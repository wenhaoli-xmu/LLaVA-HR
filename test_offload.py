import IPython
import torch
import sys, gc


def offload(x: torch.Tensor):

    rq = x.requires_grad
    x.requires_grad_(False)

    if x.grad is not None:
        tmp = x.grad
        x.grad = None
        tmp = tmp.to('cpu', non_blocking=True)
    else:
        tmp = None

    x = x.to('cpu', non_blocking=True)

    if tmp is not None:
        x.grad = tmp

    x.requires_grad_(rq)
    
    return x


def restore(x: torch.Tensor):
    if x.grad is not None:
        tmp = x.grad
        x.grad = None
        tmp = tmp.to('cuda', non_blocking=True)
    else:
        tmp = None

    x = x.to('cuda', non_blocking=True)

    if tmp is not None:
        x.grad = tmp
    
    return x


class X(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.ParameterList()

    def append(self, data, grad):
        self.value.append(torch.nn.Parameter(data, requires_grad=True))
        self.value[-1].grad = grad

    def offload(self):
        self.to('cpu')

    def restore(self):
        self.to('cuda')

x = X()
x.append(
    torch.rand(1024 * 1024 * 1024, device='cuda'), 
    torch.rand(1024 * 1024 * 1024, device='cuda')
)

torch.cuda.empty_cache()
print(f"at creation: {torch.cuda.memory_allocated()}")

x.offload()
torch.cuda.empty_cache()
print(f"after offload: {torch.cuda.memory_allocated()}")
IPython.embed(header='debug')

x.restore()
torch.cuda.empty_cache()
print(f"after restore: {torch.cuda.memory_allocated()}")
IPython.embed(header='debug')