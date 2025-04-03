import torch
import torch.distributed as dist
from chunkoptim.utils import SecoCache, chunkize
from contextlib import nullcontext
from itertools import chain

from transformers.models.llama.modeling_llama import (
    _make_causal_mask,
    _expand_mask)


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
def _sample_tokens_v2(inputs_embeds, labels, attention_mask, image_masks, chunk_budget, bwd_chunk_size, attns, raw_loss):
    text_masks = attention_mask.clone()
    text_masks[image_masks] = False
    indices = []
    valid_length = []

    TEXT_RATIO_MAX = 0.5

    text_budget = int(bwd_chunk_size * chunk_budget * TEXT_RATIO_MAX)

    for text_mask, image_mask, attn, rloss in zip(text_masks, image_masks, attns, raw_loss):
        full_ids = torch.arange(text_masks.shape[1], device='cuda')
        text_ids = full_ids[text_mask]
        image_ids = full_ids[image_mask]

        nonzero_selector = rloss[text_ids] > 0
        text_ids = text_ids[nonzero_selector]

        """这里假设给text_ids分配所有budget的一半"""
        if text_ids.numel() < text_budget:
            select_ids = text_ids
        else:
            topk_selector = torch.topk(rloss[text_ids], k=text_budget).indices
            select_ids = text_ids[topk_selector]

        """剩下的budget是image的"""
        remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
        if image_ids.numel() > 0:
            topk_selector = attn.topk(k=remain_budget).indices
            select_ids = torch.cat([select_ids, image_ids[topk_selector]], dim=0)
            remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
        valid_length.append(select_ids.numel())

        """多余的ids要通过padding来补齐"""
        if remain_budget > 0:
            pad_ids = torch.zeros((remain_budget,), dtype=select_ids.dtype, device='cuda')
            select_ids = torch.cat([select_ids, pad_ids], dim=0)
            
        indices.append(select_ids)

    indices, _ = torch.stack(indices, dim=0).sort()
    pos_ids = indices
    labels = torch.gather(labels, dim=-1, index=indices)

    for i in range(len(valid_length)):
        if valid_length[i] < chunk_budget * bwd_chunk_size:
            labels[i, valid_length[i]:] = -100

    inputs_embeds_indices = indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
    inputs_embeds = torch.gather(inputs_embeds, dim=1, index=inputs_embeds_indices)
    inputs_embeds.requires_grad_(True)

    return inputs_embeds, labels, pos_ids, indices
