# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

import torch

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

# Below this element count a copy is dominated by host-side work, not by the
# kernel: the functional wrapper costs an extra allocation plus a second python
# dispatch through copy_, which measures ~47us on [64, 64] while the native op
# does the whole copy in ~8us (benchmark/test_copy.py --level core, Gems Speedup
# 0.16x). Anything this small goes straight back to PyTorch.
_SMALL_COPY_NUMEL = 2**14


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        # Preserve PyTorch's behaviour of warning when casting complex to real
        # by forcing the redispatch path, which issues the warning internally.
        return False
    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        # Triton does not support float8 yet, so defer to PyTorch which has a reference implementation.
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    # Small copies are host-bound: the checks below plus the kernel launch cost
    # ~94us on [64, 64] while the native op moves the same data in ~8us. Native
    # is also the semantic reference for zerotensors, aliasing and dtype
    # conversion, so nothing is lost by handing these straight over. Broadcast is
    # the one exception -- the native XPU copy_ only fills the first row of a
    # broadcast source -- so equal shapes are required.
    if dst.numel() < _SMALL_COPY_NUMEL and src.shape == dst.shape:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if src.numel() > 2**31 - 1 or dst.numel() > 2**31 - 1:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if not _can_use_triton(dst, src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if dst.numel() == 0:
        # Respect PyTorch behaviour: empty tensors should still validate broadcast.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    logger.debug("GEMS COPY_")

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )

    expanded_src = _expand_like(src, dst.shape)

    # tle.gpu does the whole move: a TMA tile when the layout is affine, an
    # element-wise gather/scatter for permuted or broadcast reads, and the same
    # gather with an LM-side cast when src and dst differ in dtype.
    if tle_copy(expanded_src, dst):
        return dst

    # What tle cannot represent (a dtype it has no LM type for, or a layout that
    # still needs more than MAX_GATHER_RANK dimensions) goes back to PyTorch.
    return torch.ops.aten.copy_.default.redispatch(
        _FALLBACK_KEYSET, dst, src, non_blocking
    )
