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

"""`tle.gpu` implementation of the pure-copy operators.

Two kernels cover every copy:

* `_tle_tile_copy_kernel` -- both sides contiguous, same dtype: one flat TMA
  tile GM -> LM -> GM. This is the throughput path (1.5-1.7x the pointwise
  kernel up to ~16M elements on KL3, level with `aten::copy_` above that).
* `_tle_gather_copy_kernel` -- everything else. Handing `tle.gpu.copy` a GM
  pointer tensor instead of a descriptor makes it lower to an element-wise
  gather (g2l) / scatter (l2g), which is what permuted, broadcast, strided and
  overlapping reads need; with `CONVERT` it also casts inside LM, so a `copy_`
  that changes dtype stays on the same kernel.

`tle_copy` returns False only when tle has no LM type for the dtype, or when the
layout still needs more than `MAX_GATHER_RANK` dimensions after collapsing. The
caller then keeps its own kernel.
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

# tritonxpu-tle-core-tiling hands each of the 64 cores whole rows of the tile.
CORE_NUM = 64
# LM per core for the tile path. triton/docs/xpu3/tle_user_guide.md quotes ~2KB
# as the budget and test/tle/test_tle_copy_1d.py sizes its block that way, but
# 4KB still compiles and is measurably better on large tensors (fp16 2**28:
# 630us -> 577us, i.e. level with aten::copy_); 8KB fails to allocate.
LM_BYTES_PER_CORE = 4096
# Elements per program on the gather path. That path materializes a vector of GM
# pointers, so the block is bounded by the vector budget rather than by LM:
# 16384 still compiles, 32768 fails with "Failed to tune buffer size".
GATHER_BLOCK = 8192
# Dimensions the gather kernel unrolls; deeper layouts collapse or fall back.
MAX_GATHER_RANK = 5

# dtypes verified on KL3. torch.bool is absent on purpose: i1 is packed in LM and
# comes back corrupted, so bool tensors move through an int8 view (_byte_view).
_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}


@triton.jit
def _tle_tile_copy_kernel(src_desc, dst_desc, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
    """Flat TMA tile: the descriptor lowering clamps the tail block itself."""
    pid = tl.program_id(0)
    buf = tle.gpu.alloc([BLOCK], dtype=DTYPE, layout=None, scope=tle.gpu.lmem)
    tle.gpu.copy(src_desc, buf, [BLOCK], [pid * BLOCK])
    tle.gpu.copy(buf, dst_desc, [BLOCK], [pid * BLOCK])


@triton.jit
def _tle_gather_copy_kernel(
    src_ptr,
    dst_ptr,
    numel,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    t0,
    t1,
    t2,
    t3,
    t4,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    SRC_DTYPE: tl.constexpr,
    DST_DTYPE: tl.constexpr,
    CONVERT: tl.constexpr,
    TO_BOOL: tl.constexpr,
):
    """Element-wise copy through LM, for layouts a descriptor cannot express.

    `d*` / `s*` / `t*` are the extents and the src/dst strides, innermost
    dimension first, with unused dimensions carrying extent 1 and stride 0.

    The gather takes no mask, so the tail is clamped rather than masked: the last
    lanes redo an element that is already theirs, and because both offsets come
    from the same clamped index the repeated write stores the same value.

    With CONVERT the value is cast in LM (`local_ptr` + `tl.load`/`tl.store`),
    which is how a dtype-changing `copy_` is served; TO_BOOL reproduces
    PyTorch's "nonzero becomes True" rule.
    """
    pid = tl.program_id(0)
    idx = tl.minimum(pid * BLOCK + tl.arange(0, BLOCK), numel - 1)

    rem = idx
    src_off = idx * 0
    dst_off = idx * 0
    i = rem % d0
    rem = rem // d0
    src_off += i * s0
    dst_off += i * t0
    if RANK > 1:
        i = rem % d1
        rem = rem // d1
        src_off += i * s1
        dst_off += i * t1
    if RANK > 2:
        i = rem % d2
        rem = rem // d2
        src_off += i * s2
        dst_off += i * t2
    if RANK > 3:
        i = rem % d3
        rem = rem // d3
        src_off += i * s3
        dst_off += i * t3
    if RANK > 4:
        i = rem % d4
        src_off += i * s4
        dst_off += i * t4

    src_buf = tle.gpu.alloc([BLOCK], dtype=SRC_DTYPE, layout=None, scope=tle.gpu.lmem)
    tle.gpu.copy(src_ptr + src_off, src_buf, [BLOCK])
    if CONVERT:
        dst_buf = tle.gpu.alloc(
            [BLOCK], dtype=DST_DTYPE, layout=None, scope=tle.gpu.lmem
        )
        lane = tl.arange(0, BLOCK)
        val = tl.load(tle.gpu.local_ptr(src_buf, (lane,)))
        if TO_BOOL:
            val = (val != 0).to(DST_DTYPE)
        else:
            val = val.to(DST_DTYPE)
        tl.store(tle.gpu.local_ptr(dst_buf, (lane,)), val)
        tle.gpu.copy(dst_buf, dst_ptr + dst_off, [BLOCK])
    else:
        tle.gpu.copy(src_buf, dst_ptr + dst_off, [BLOCK])


def tle_dma_available():
    """tle.gpu exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


def _byte_view(t: torch.Tensor) -> torch.Tensor:
    """bool has no LM representation (i1 is packed), so move it as int8."""
    return t.view(torch.int8) if t.dtype == torch.bool else t


def _collapse(shape, src_stride, dst_stride):
    """Fold the copy into as few dimensions as the two stride sets allow.

    Returns (dims, src_strides, dst_strides) innermost-first and padded to
    MAX_GATHER_RANK, plus the real rank; None if it does not fit.
    """
    dims = [
        (int(n), int(ss), int(ds))
        for n, ss, ds in zip(shape, src_stride, dst_stride)
        if n != 1
    ]
    if not dims:
        dims = [(1, 0, 0)]
    # Innermost = fastest varying in the destination, so the scatter stays as
    # close to sequential as the layout allows.
    dims.sort(key=lambda dim: abs(dim[2]))
    merged = []
    for n, ss, ds in dims:
        if merged:
            pn, pss, pds = merged[-1]
            if ss == pss * pn and ds == pds * pn:
                merged[-1] = (pn * n, pss, pds)
                continue
        merged.append((n, ss, ds))
    rank = len(merged)
    if rank > MAX_GATHER_RANK:
        return None
    pad = [(1, 0, 0)] * (MAX_GATHER_RANK - rank)
    merged += pad
    return (
        [m[0] for m in merged],
        [m[1] for m in merged],
        [m[2] for m in merged],
        rank,
    )


def tle_copy(src: torch.Tensor, dst: torch.Tensor) -> bool:
    """Copy `src` into `dst` with tle.gpu; False if tle cannot express it."""
    if not tle_dma_available():
        return False
    if src.device != dst.device or src.numel() != dst.numel() or src.numel() == 0:
        return False

    src_v, dst_v = _byte_view(src), _byte_view(dst)
    src_ty, dst_ty = _TL_DTYPE.get(src_v.dtype), _TL_DTYPE.get(dst_v.dtype)
    if src_ty is None or dst_ty is None:
        return False

    numel = src.numel()
    convert = src.dtype != dst.dtype

    if not convert and src_v.is_contiguous() and dst_v.is_contiguous():
        block = LM_BYTES_PER_CORE // src_v.element_size() * CORE_NUM
        _tle_tile_copy_kernel[(triton.cdiv(numel, block),)](
            TensorDescriptor.from_tensor(src_v.view(-1), [block]),
            TensorDescriptor.from_tensor(dst_v.view(-1), [block]),
            block,
            src_ty,
        )
        logger.debug("GEMS_KUNLUNXIN TLE_COPY tile numel=%d", numel)
        return True

    if src.shape != dst.shape:
        return False
    collapsed = _collapse(src.shape, src.stride(), dst.stride())
    if collapsed is None:
        return False
    dims, src_strides, dst_strides, rank = collapsed

    _tle_gather_copy_kernel[(triton.cdiv(numel, GATHER_BLOCK),)](
        src_v,
        dst_v,
        numel,
        *dims,
        *src_strides,
        *dst_strides,
        rank,
        GATHER_BLOCK,
        src_ty,
        dst_ty,
        convert,
        dst.dtype == torch.bool,
    )
    logger.debug(
        "GEMS_KUNLUNXIN TLE_COPY gather numel=%d rank=%d convert=%s",
        numel,
        rank,
        convert,
    )
    return True
