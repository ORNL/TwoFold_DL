import torch
from typing import Optional, Tuple, Any, List
from types import FunctionType

@torch.jit.script
def dynamic_length_slice(
    x: torch.Tensor, start: int = 0, size: int = 1024
) -> torch.Tensor:
    """Slices a tensor along the second axis.
    Ex: (b n h d) -> (b n[start:start+size] h d)
    """
    # avoid slicing overhead if not needed
    if start == 0 and start + size >= x.shape[1]:
        return x
    else:
        return x[:, start : start + size]


@torch.jit.script
def dynamic_slice(
    x: torch.Tensor,
    start: Tuple[int, int, int],
    slice_sizes: Tuple[int, int, int],
) -> torch.Tensor:
    """approx like jax.lax.dynamic_slice.
    * NOTE: assumes we dont work on first dim
    Ex:
    dynamic_slice(
        x,
        slices=(0, 0, 0),
        slice_sizes=(16, 64, 64)
    )
    """
    return x[
        :,
        start[0] : start[0] + slice_sizes[0],
        start[1] : start[1] + slice_sizes[1],
        start[2] : start[2] + slice_sizes[2],
    ]


def torch_map(fn, xs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """approx like jax.lax.map"""
    return


def torch_scan(
    f: FunctionType, init: int = 0, xs: Optional[List] = None, length: int = 0
):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    if len(ys) > 0 and isinstance(ys[0], tuple):
        return carry, tuple((torch.cat([y[i] for y in ys], dim=1) for i in range(len(ys[0]))))

    return carry, torch.cat(ys, dim=1)
