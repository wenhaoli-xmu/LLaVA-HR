import types
import torch
import torch.distributed as dist

from functools import partial
from chunkoptim.utils import SecoCache, chunkize
from contextlib import nullcontext


def reset_training_step(trainer):
    func_mapper = {
        'v1': _training_step_seco}
    if hasattr(trainer.model, "modify_version") and trainer.model.modify_version in func_mapper:
        step_func = func_mapper[trainer.model.modify_version]
        trainer.training_step = types.MethodType(step_func, trainer)


def _set_to_incomming_grad(_, incomming_grad):
    return incomming_grad


def _training_step_parallel(self, model, inputs):
    valid_label_count = (inputs['labels'] != -100).sum()
    seco_cache = SecoCache(len(model.model.layers))

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        (
            _, 
            attention_mask, 
            _, 
            inputs_embeds, 
            labels, 
            _
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            past_key_values=None,
            labels=inputs['labels'],
            images=inputs['images'])

    outputs = model(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        past_key_values=seco_cache)

    loss = outputs['loss'].sum() / valid_label_count
    self.accelerator.backward(loss)

    return loss.detach() / self.args.gradient_accumulation_steps


def _backward(model, loss):
    if dist.is_initialized():
        model.backward(loss)
    else:
        loss.backward()


def _maybe_align_length_across_gpus(inputs_embeds, labels, attention_mask):
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
            labels = torch.cat([
                labels, 
                torch.full((bsz, pad_length), fill_value=-100, dtype=torch.int64, device='cuda')],
                dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.full((bsz, pad_length), fill_value=False, dtype=torch.bool, device='cuda')],
                dim=1)
            
    return inputs_embeds, labels, attention_mask



def _first_forward_prop_seco(model, inputs_embeds_list, labels_list, masks_list, valid_label_count):
    accum_loss = torch.tensor(0, dtype=inputs_embeds_list[0].dtype, device='cuda')
    num_layers = len(model.model.layers)
    seco_cache = SecoCache(num_layers)

    with torch.no_grad():
        for i, (chunk_embeds, chunk_labels) in enumerate(zip(inputs_embeds_list, labels_list)):
            outputs = model(
                input_ids=None,
                attention_mask=torch.cat(masks_list[:i+1], dim=1),
                inputs_embeds=chunk_embeds,
                labels=chunk_labels,
                past_key_values=seco_cache,
                shift_label=False)
            accum_loss += outputs['loss'].sum() / valid_label_count

    return accum_loss, seco_cache



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
            _
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            past_key_values=None, 
            labels=inputs['labels'],
            images=inputs['images'])

    return attention_mask, inputs_embeds, labels


def _training_step_seco(self, model, inputs):
    fwd_chunk_size = 64
    bwd_chunk_size = 64
    valid_label_count = (inputs['labels'] != -100).sum()


    # prepare inputs
    attention_mask, inputs_embeds, labels = _prepare_inputs(model, inputs, no_grad=True)
    origin_seq_len = inputs_embeds.shape[-2]

    # align length across gpus
    inputs_embeds, labels, attention_mask = _maybe_align_length_across_gpus(
        inputs_embeds, 
        labels, 
        attention_mask)

    inputs_embeds_detach = inputs_embeds.detach()
    inputs_embeds_detach.requires_grad_(True)
    labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

    # chunkize inputs
    inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
        [inputs_embeds_detach, labels, attention_mask],
        chunk_size=fwd_chunk_size)
    
    # first forward propagation
    accum_loss, seco_cache = _first_forward_prop_seco(
        model, 
        inputs_embeds_list, 
        labels_list, 
        masks_list, 
        valid_label_count)

    # maybe change chunk size before backward prop
    if fwd_chunk_size != bwd_chunk_size:
        seco_cache.reorganize(bwd_chunk_size)
        inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
            [inputs_embeds_detach, labels, attention_mask],
            chunk_size=bwd_chunk_size,)

    # seco backward prop
    generator = reversed(list(enumerate(zip(inputs_embeds_list, labels_list))))
    for i, (chunk_embeds, chunk_labels) in generator:
        tmp_cache = seco_cache.range(i)
        outputs = model(
            input_ids=None,
            attention_mask=torch.cat(masks_list[:i+1], dim=1),
            inputs_embeds=chunk_embeds,
            labels=chunk_labels,
            past_key_values=tmp_cache,
            shift_label=False)
        loss = outputs['loss'].sum() / valid_label_count
        tmp_cache.index(i).copy_scaled_grad(gd=seco_cache.index(i).grad)
        _backward(model, loss)

    # second forward for vision tower
    _, inputs_embeds, _ = _prepare_inputs(model, inputs)

    # vision tower backward propagation
    if dist.is_initialized():
        gradient = inputs_embeds_detach.grad[:,:origin_seq_len]
    else:
        gradient = inputs_embeds_detach.grad
    inputs_embeds.register_hook(partial(
        _set_to_incomming_grad, 
        incomming_grad=gradient))
    _backward(model, inputs_embeds.sum())

    # optimizer step
    if dist.is_initialized():
        model.step()
    else:
        # TODO
        raise NotImplementedError

    return accum_loss / self.args.gradient_accumulation_steps
