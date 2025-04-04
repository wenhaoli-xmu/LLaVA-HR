import torch
import torch.distributed as dist
from chunkoptim.utils import SecoCache, chunkize
from contextlib import nullcontext
from itertools import chain

from transformers.models.llama.modeling_llama import (
    _make_causal_mask,
    _expand_mask)


from transformers import Trainer


def _construct_4d_mask(attention_mask, dtype, device):
    bsz, seq_len = attention_mask.shape
    mask = _make_causal_mask(
        (bsz, seq_len),
        dtype=dtype,
        device=device,
        past_key_values_length=0)
    return mask


def _set_to_incomming_grad(base_grad, incomming_grad, indices=None):
    if indices is None:
        return incomming_grad
    else:
        base_grad.zero_()
        indices = indices.unsqueeze(-1).expand(-1, -1, base_grad.shape[-1])
        return torch.scatter(base_grad, dim=1, index=indices, src=incomming_grad)


def _step(model, optimizer):
    if dist.is_initialized():
        model.step()
    else:
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.empty_cache()


def _backward(loss, model):
    if dist.is_initialized():
        # deepspeed
        model.backward(loss)
    else:
        loss.backward()


@torch.no_grad()
def _first_forward_prop_seco(model, inputs_embeds_list, labels_list, attention_mask, valid_label_count, return_raw_loss=False):
    
    if not return_raw_loss:
        accum_loss = torch.tensor(0, dtype=inputs_embeds_list[0].dtype, device='cuda')
    else:
        raw_loss = []

    num_layers = len(model.model.layers)
    seco_cache = SecoCache(num_layers)
    batch_size, chunk_size = inputs_embeds_list[0].shape[:2]

    for i, (chunk_embeds, chunk_labels) in enumerate(zip(inputs_embeds_list, labels_list)):
        if isinstance(attention_mask, list):
            mask = torch.cat(attention_mask[:i+1], dim=1)
        else:
            start = i * chunk_size
            end = (i + 1) * chunk_size
            mask = attention_mask[..., start: end, :end]

        outputs = model(
            attention_mask=mask,
            inputs_embeds=chunk_embeds,
            labels=chunk_labels,
            past_key_values=seco_cache,
            shift_label=False,
            is_reduce=False)
        
        if not return_raw_loss:
            accum_loss += outputs['loss'].sum() / valid_label_count
        else:
            raw_loss.append(outputs['loss'])

    return torch.cat(raw_loss, dim=-1).reshape(batch_size, -1) if return_raw_loss else accum_loss, seco_cache


def _chunkize_inputs(tensor_list, chunk_size, dim=1):
    output_list = []
    for x in tensor_list:
        x = list(chunkize(x, dim=dim, chunk_size=chunk_size))
        output_list.append(x)
    return output_list


def _prepare_inputs(model, inputs, no_grad=False):

    no_grad_context = torch.no_grad() if no_grad else nullcontext()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16), no_grad_context:
        (
            _, 
            attention_mask, 
            _, 
            inputs_embeds, 
            labels,
            image_masks,
            attns
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            past_key_values=None, 
            labels=inputs['labels'],
            images=inputs['images'])

    return attention_mask, inputs_embeds, labels, image_masks, attns


@torch.no_grad()
def _sample_tokens_v2(inputs_embeds, labels, attention_mask, image_masks, chunk_budget, bwd_chunk_size, attns):
    text_masks = attention_mask.clone()
    text_masks[image_masks] = False
    indices = []
    sparse_labels = []

    TEXT_RATIO_MAX = 0.5

    text_budget = int(bwd_chunk_size * chunk_budget * TEXT_RATIO_MAX)

    for text_mask, image_mask, attn, label in zip(text_masks, image_masks, attns, labels):
        full_ids = torch.arange(text_masks.shape[1], device='cuda')
        text_ids = full_ids[text_mask]
        image_ids = full_ids[image_mask]
        valid_label_mask = label != -100

        # nonzero_selector = label[text_ids] > 0
        # text_ids = text_ids[nonzero_selector]

        """这里假设给text_ids分配所有budget的一半"""
        if text_ids.numel() < text_budget:
            select_ids = text_ids
        else:
            arange = torch.arange(label.numel(), device=label.device)
            valid_label_ids = arange[valid_label_mask]

            if valid_label_ids.numel() > text_budget:
                selector = torch.randperm(valid_label_ids.numel())[:text_budget]
                select_ids = valid_label_ids[selector]
            else:
                select_ids = valid_label_ids

            remain_budget = text_budget - select_ids.numel()
            if remain_budget > 0:
                is_text_mask = torch.zeros_like(valid_label_mask).scatter_(dim=-1, index=text_ids, value=True)
                valid_ids = arange[~valid_label_mask & is_text_mask]
                selector = torch.randperm(valid_ids.numel())[:remain_budget]
                select_ids = torch.cat([select_ids, valid_ids[selector]], dim=0)

            # topk_selector = torch.topk(label[text_ids], k=text_budget).indices
            # select_ids = text_ids[topk_selector]
            
            # selector = torch.randperm(text_ids.numel(), device=text_ids.device)[:text_budget]
            # select_ids = text_ids[selector]

        """剩下的budget是image的"""
        remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
        if image_ids.numel() > 0:
            topk_selector = attn.topk(k=remain_budget).indices
            select_ids = torch.cat([select_ids, image_ids[topk_selector]], dim=0)
            remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
            
            # selector = torch.randperm(image_ids.numel(), device='cuda')[:remain_budget]
            # select_ids = torch.cat([select_ids, image_ids[selector]])
            # remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()


        select_ids = select_ids.sort()
        sparse_label = label[select_ids]

        """多余的ids要通过padding来补齐"""
        if remain_budget > 0:
            pad_ids = torch.zeros((remain_budget,), dtype=torch.int32, dtype=select_ids.dtype, device='cuda')
            pad_label = torch.full((remain_budget,), fill_value=-100, dtype=torch.int32, device='cuda')
            select_ids = torch.cat([select_ids, pad_ids])
            sparse_label = torch.cat([sparse_label, pad_label])

        sparse_label = torch.gather(sparse_label, dim=-1, index=selector)
        indices.append(select_ids)
        sparse_labels.append(sparse_label)

    indices = torch.stack(indices)
    sparse_labels = torch.stack(sparse_labels)
    pos_ids = indices

    inputs_embeds_indices = indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
    inputs_embeds = torch.gather(inputs_embeds, dim=1, index=inputs_embeds_indices)
    inputs_embeds.requires_grad_(True)

    return inputs_embeds, sparse_labels, pos_ids, indices
