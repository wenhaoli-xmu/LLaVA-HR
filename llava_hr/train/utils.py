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


def _maybe_align_length_across_gpus(inputs_embeds, labels, attention_mask, image_masks):
    if dist.is_initialized():
        # gather length & calculate maximum length
        bsz, seq_len, hidden_size = inputs_embeds.shape
        length_pool = [torch.tensor(0, device='cuda') for _ in range(dist.get_world_size())]
        dist.all_gather(length_pool, torch.tensor(seq_len, device='cuda'))
        max_length = max([x.item() for x in length_pool])
        pad_length = max_length - seq_len

        # pad input tensors
        if pad_length > 0:
            inputs_embeds = torch.cat([
                inputs_embeds, 
                torch.zeros((bsz, pad_length, hidden_size), dtype=inputs_embeds.dtype, device='cuda')],
                dim=1)
            
            if labels is not None:
                labels = torch.cat([
                    labels, 
                    torch.full((bsz, pad_length), fill_value=-100, dtype=torch.int64, device='cuda')],
                    dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.full((bsz, pad_length), fill_value=False, dtype=torch.bool, device='cuda')],
                    dim=1)
            
            if image_masks is not None:
                image_masks = torch.cat([
                    image_masks, 
                    torch.full((bsz, pad_length), fill_value=False, device='cuda')],
                    dim=1)
            
    return inputs_embeds, labels, attention_mask, image_masks


@torch.no_grad()
def _first_forward_prop_seco(model, inputs_embeds_list, labels_list, attention_mask, valid_label_count, cpu_offload=None):
    accum_loss = torch.tensor(0, dtype=inputs_embeds_list[0].dtype, device='cuda')
    num_layers = len(model.model.layers)
    seco_cache = SecoCache(num_layers, cpu_offload)
    chunk_size = inputs_embeds_list[0].shape[1]

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
        accum_loss += outputs['loss'].sum() / valid_label_count

    return accum_loss, seco_cache


@torch.no_grad()
def _first_forward_prop_seco2(model, inputs_embeds_list, attention_mask):
    num_layers = len(model.model.layers)
    seco_cache = SecoCache(num_layers)
    chunk_size = inputs_embeds_list[0].shape[1]

    for i, chunk_embeds in enumerate(inputs_embeds_list):
        if isinstance(attention_mask, list):
            mask = torch.cat(attention_mask[:i+1], dim=1)
        else:
            start = i * chunk_size
            end = (i + 1) * chunk_size
            mask = attention_mask[..., start: end, :end]
        
        model(
            attention_mask=mask,
            inputs_embeds=chunk_embeds,
            past_key_values=seco_cache,
            shift_label=False)

    return seco_cache


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
            image_masks
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            past_key_values=None, 
            labels=inputs['labels'],
            images=inputs['images'])

    return attention_mask, inputs_embeds, labels, image_masks


def _sample_chunks(attention_mask, image_masks, bwd_chunk_size, chunk_budget):

    # assign text/image quota
    text_masks = attention_mask.clone()
    text_masks[image_masks] = False

    text_masks_list, image_masks_list = _chunkize_inputs(
        [text_masks, image_masks], 
        chunk_size=bwd_chunk_size)
    
    has_text = [x.count_nonzero().item() > 0 for x in text_masks_list]
    has_image = [x.count_nonzero().item() > 0 for x in image_masks_list]

    num_text_chunks = sum(has_text)
    num_image_chunks = sum(has_image)

    text_budget = chunk_budget // 2
    image_budget = chunk_budget // 2

    if text_budget > num_text_chunks:
        image_budget += text_budget - num_text_chunks
        text_budget = num_text_chunks
    
    if image_budget > num_image_chunks:
        text_budget += image_budget - num_image_chunks
        image_budget = num_image_chunks

    num_chunks = len(has_text)
    indices = torch.arange(num_chunks, dtype=torch.int64)

    # select image chunks randomly
    image_chunk_mask = torch.tensor(has_image, dtype=torch.bool)
    image_chunk_indices = indices[image_chunk_mask]
    image_sample_indices = torch.gather(
        image_chunk_indices, 
        dim=0, 
        index=torch.randperm(image_chunk_indices.numel())[:image_budget]
        ).tolist()
    
    # select text chunks randomly
    text_chunk_mask = torch.tensor(has_text, dtype=torch.bool)
    text_chunk_mask[image_sample_indices] = False
    text_chunk_indices = indices[text_chunk_mask]
    text_sample_indices = torch.gather(
        text_chunk_indices,
        dim=0,
        index=torch.randperm(text_chunk_indices.numel())[:text_budget]
        ).tolist()
        
    sample_indices = text_sample_indices + image_sample_indices
    sample_indices = sorted(sample_indices)
    return sample_indices


def _reorganize_list(x, dim1, dim2):
    y = []
    for i in range(dim1):
        y.append([])
        for j in range(dim2):
            y[-1].append(x[i * dim2 + j])
    return y


@torch.no_grad()
def _sample_tokens(inputs_embeds, labels, attention_mask, image_masks, chunk_budget, bwd_chunk_size):
    assert bwd_chunk_size % 2 == 0
    text_masks = attention_mask.clone()
    text_masks[image_masks] = False
    indices = []
    valid_length = []

    for text_mask, image_mask in zip(text_masks, image_masks):
        full_ids = torch.arange(text_masks.shape[1], device='cuda')
        text_ids = full_ids[text_mask]
        image_ids = full_ids[image_mask]

        if text_ids.numel() < chunk_budget // 2 * bwd_chunk_size:
            select_ids = text_ids
        else:
            selector = torch.randperm(text_ids.numel(), device='cuda')
            selector = selector[:chunk_budget // 2 * bwd_chunk_size]
            select_ids = text_ids[selector]
        
        remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
        selector = torch.randperm(image_ids.numel(), device='cuda')[:remain_budget]
        select_ids = torch.cat([select_ids, image_ids[selector]], dim=0)
        valid_length.append(select_ids.numel())

        remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
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


@torch.no_grad()
def _sample_tokens_v2(inputs_embeds, labels, attention_mask, image_masks, chunk_budget, bwd_chunk_size):
    text_masks = attention_mask.clone()
    text_masks[image_masks] = False
    indices = []
    valid_length = []

    TEXT_RATIO_MAX = 0.5
    text_budget = int(bwd_chunk_size * chunk_budget * TEXT_RATIO_MAX)

    for text_mask, image_mask in zip(text_masks, image_masks):
        full_ids = torch.arange(text_masks.shape[1], device='cuda')
        text_ids = full_ids[text_mask]
        image_ids = full_ids[image_mask]

        """这里假设给text_ids分配所有budget的一半"""
        if text_ids.numel() < text_budget:
            select_ids = text_ids
        else:
            selector = torch.randperm(text_ids.numel(), device='cuda')
            selector = selector[:text_budget]
            select_ids = text_ids[selector]

        """剩下的budget是image的"""
        remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
        selector = torch.randperm(image_ids.numel(), device='cuda')[:remain_budget]
        select_ids = torch.cat([select_ids, image_ids[selector]], dim=0)
        valid_length.append(select_ids.numel())

        remain_budget = chunk_budget * bwd_chunk_size - select_ids.numel()
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
