import types
import torch
import torch.distributed as dist

from functools import partial
from chunkoptim.utils import SecoCache, chunkize


def maybe_modify_training_method(trainer):

    """
    modify version -> training function

    v0 -> none
    v1 -> seco
    
    """

    if hasattr(trainer.model, "modify_version"):
        if trainer.model.modify_version == 'v1':
            trainer.training_step = types.MethodType(training_step_seco, trainer)


def training_step_seco(self, model, inputs):

    chunk_size = 64
    valid_label_count = (inputs['labels'] != -100).sum()

    # vision encoder forward prop
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, attention_mask, _, inputs_embeds, labels, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            past_key_values=None,
            labels=inputs['labels'],
            images=inputs['images'])
        

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

    inputs_embeds_detach = inputs_embeds.detach()
    inputs_embeds_detach.requires_grad_(True)

    inputs_embeds_list = list(chunkize(inputs_embeds_detach, dim=1, chunk_size=chunk_size))
    labels_list = list(chunkize(labels, dim=1, chunk_size=chunk_size))
    attention_mask = list(chunkize(attention_mask, dim=1, chunk_size=chunk_size))
    # image_masks = list(chunkize(torch.tensor(image_masks), dim=1, chunk_size=chunk_size))  

    num_layers = len(model.model.layers)
    seco_cache = SecoCache(num_layers)

    accum_loss = torch.tensor(0, dtype=inputs_embeds.dtype, device='cuda')

    # forward propagation
    with torch.no_grad():
        for i, (chunk_embeds, chunk_labels) in enumerate(zip(inputs_embeds_list, labels_list)):
        
            outputs = model(
                input_ids=None,
                attention_mask=torch.cat(attention_mask[:i+1], dim=1),
                inputs_embeds=chunk_embeds,
                labels=chunk_labels,
                past_key_values=seco_cache)
            
            accum_loss += outputs['loss'].sum() / valid_label_count

    generator = reversed(list(enumerate(zip(inputs_embeds_list, labels_list))))

    # backward propagation
    for i, (chunk_embeds, chunk_labels) in generator:

        tmp_cache = seco_cache.range(i)

        outputs = model(
            input_ids=None,
            attention_mask=torch.cat(attention_mask[:i+1], dim=1),
            inputs_embeds=chunk_embeds,
            labels=chunk_labels,
            past_key_values=tmp_cache)
        
        loss = outputs['loss'].sum() / valid_label_count

        tmp_cache.index(i).copy_scaled_grad(gd=seco_cache.index(i).grad)
        self.accelerator.backward(loss)

    def set_to_incomming_grad(base_grad, incomming_grad):
        return incomming_grad
    
    # scaling gradient and loss
    accum_loss.div_(self.args.gradient_accumulation_steps)
    inputs_embeds_detach.grad.div_(self.args.gradient_accumulation_steps)
    
    inputs_embeds.register_hook(partial(
        set_to_incomming_grad, 
        incomming_grad=inputs_embeds_detach.grad))

    self.accelerator.backward(inputs_embeds.sum())

    return accum_loss