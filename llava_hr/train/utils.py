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
            raw_loss.append(outputs['loss'].reshape(batch_size, -1))

    return torch.cat(raw_loss, dim=-1) if return_raw_loss else accum_loss, seco_cache


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


        select_ids = select_ids.sort().values
        sparse_label = label[select_ids]

        """多余的ids要通过padding来补齐"""
        if remain_budget > 0:
            pad_ids = torch.zeros((remain_budget,), dtype=select_ids.dtype, device='cuda')
            pad_label = torch.full((remain_budget,), fill_value=-100, dtype=sparse_label.dtype, device='cuda')
            select_ids = torch.cat([select_ids, pad_ids])
            sparse_label = torch.cat([sparse_label, pad_label])

        indices.append(select_ids)
        sparse_labels.append(sparse_label)

    indices = torch.stack(indices)
    sparse_labels = torch.stack(sparse_labels)
    pos_ids = indices

    inputs_embeds_indices = indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
    inputs_embeds = torch.gather(inputs_embeds, dim=1, index=inputs_embeds_indices)
    inputs_embeds.requires_grad_(True)

    return inputs_embeds, sparse_labels, pos_ids, indices


@torch.no_grad()
def _sample_tokens_v3(inputs_embeds, labels, attention_mask, image_masks, image_budget, attns):
    indices = []
    sparse_labels = []

    # 首先找到所有instance中label数量最多的
    num_labels_max = (labels != -100).sum(-1).max().item()
    if dist.is_initialized():
        local_labels_max = torch.tensor(num_labels_max, device='cuda')
        global_labels_max = [torch.empty_like(local_labels_max) for _ in range(dist.get_world_size())]
        dist.all_gather(global_labels_max, local_labels_max)
        num_labels_max = max([x.item() for x in global_labels_max])

    # 总的budget等于image部分加上labels部份
    num_budget = num_labels_max + image_budget

    for image_mask, attn, label in zip(image_masks, attns, labels):

        remain_budget = num_budget

        # 找到image部分的indices
        full_ids = torch.arange(label.numel(), device='cuda')
        image_ids = full_ids[image_mask]

        # 首先将所有的labels部分选择
        valid_label_mask = label != -100
        arange = torch.arange(label.numel(), dtype=torch.int64, device='cuda')
        select_ids = arange[valid_label_mask]
        remain_budget -= select_ids.numel()

        if image_ids.numel() > 0:
            # 如果有图片，则图片部分选择image budget个token，根据attention分数来选择
            topk_selector = attn.topk(k=image_budget).indices
            select_ids = torch.cat([select_ids, image_ids[topk_selector]], dim=0)
            remain_budget -= topk_selector.numel()

        # 选择好了之后排序，并且获取标签
        select_ids = select_ids.sort().values
        sparse_label = label[select_ids]

        # 剩余的部分需要通过padding来对其
        if remain_budget > 0:
            pad_ids = torch.zeros((remain_budget,), dtype=select_ids.dtype, device='cuda')
            pad_label = torch.full((remain_budget,), fill_value=-100, dtype=sparse_label.dtype, device='cuda')
            select_ids = torch.cat([select_ids, pad_ids])
            sparse_label = torch.cat([sparse_label, pad_label])

        indices.append(select_ids)
        sparse_labels.append(sparse_label)

    indices = torch.stack(indices)
    sparse_labels = torch.stack(sparse_labels)
    pos_ids = indices

    inputs_embeds_indices = indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
    inputs_embeds = torch.gather(inputs_embeds, dim=1, index=inputs_embeds_indices)
    inputs_embeds.requires_grad_(True)

    return inputs_embeds, sparse_labels, pos_ids, indices


@torch.no_grad()
def _sample_tokens_v4(inputs_embeds, labels, attention_mask, image_masks, image_budget, attns):
    indices = []
    sparse_labels = []

    # 首先找到所有instance中label数量最多的
    num_txt_max = (~image_masks).sum(-1).max().item()
    if dist.is_initialized():
        local_max = torch.tensor(num_txt_max, device='cuda')
        global_max = [torch.empty_like(local_max) for _ in range(dist.get_world_size())]
        dist.all_gather(global_max, local_max)
        num_txt_max = max([x.item() for x in global_max])

    # 总的budget等于image部分加上labels部份，其中32代表的是sink tokens
    num_budget = num_txt_max + image_budget

    for image_mask, attn, label in zip(image_masks, attns, labels):

        # 选择所有的文字
        mask = ~image_mask

        # 选择图片部分
        full_ids = torch.arange(mask.numel(), device='cuda')
        image_ids = full_ids[image_mask]
        if image_ids.numel() > 0:
            # 如果有图片，则图片部分选择image budget个token，根据attention分数来选择
            s = attn.topk(k=image_budget).indices
            mask[s] = True

        remain_budget = num_budget - mask.count_nonzero()
        select_ids = full_ids[mask]

        # 选择好了之后排序，并且获取标签
        select_ids = select_ids.sort().values
        sparse_label = label[select_ids]

        # 剩余的部分需要通过padding来对其
        if remain_budget > 0:
            pad_ids = torch.zeros((remain_budget,), dtype=select_ids.dtype, device='cuda')
            pad_label = torch.full((remain_budget,), fill_value=-100, dtype=sparse_label.dtype, device='cuda')
            select_ids = torch.cat([select_ids, pad_ids])
            sparse_label = torch.cat([sparse_label, pad_label])

        indices.append(select_ids)
        sparse_labels.append(sparse_label)

    indices = torch.stack(indices)
    sparse_labels = torch.stack(sparse_labels)
    pos_ids = indices

    inputs_embeds_indices = indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
    inputs_embeds = torch.gather(inputs_embeds, dim=1, index=inputs_embeds_indices)
    inputs_embeds.requires_grad_(True)

    return inputs_embeds, sparse_labels, pos_ids, indices