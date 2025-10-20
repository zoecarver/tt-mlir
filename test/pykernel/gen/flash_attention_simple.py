# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.d2m_api import *
from utils import assert_pcc
import torch
import math


@pykernel_gen(
    grid=(1, 1),  # Single core to debug synchronization issues first
    block_factors=[
        (1, 1),  # Q: 1x1 tile for now (single tile simplifies DMA)
        (1, 1),  # K: 1x1 tile
        (1, 1),  # V: 1x1 tile
        (1, 1),  # out: 1x1 tile
    ],
)
def flash_attention_simple(Q, K, V, out, block_factors=None, grid=None):
    # Simplified FlashAttention for Tenstorrent - gen3 API
    # Implements basic attention: softmax(Q @ K^T / sqrt(d)) @ V
    # NOTE: Uses 2D tensors (seq_len, d_head) - framework doesn't handle 4D yet
    # Without fancy optimizations - just correct semantics
    assert block_factors is not None
    assert grid is not None

    # Block configuration - only use block_factors and grid
    # DO NOT access tensor shapes - causes capture errors
    Q_BLOCK = block_factors[0][0]  # tiles
    K_BLOCK = block_factors[1][0]
    V_BLOCK = block_factors[2][0]
    OUT_BLOCK = block_factors[3][0]

    GY = grid[0]
    GX = grid[1]

    # Number of K/V blocks to process
    NUM_KV_BLOCKS = 1  # Start with single iteration to debug

    # Create streams for inputs only (output handled automatically by framework)
    Q_stream = Stream(Q)
    K_stream = Stream(K)
    V_stream = Stream(V)

    @compute()
    async def attention_compute(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        # FIXME: Need to initialize accumulator state for online softmax
        # Should have: O_acc = zeros, m_old = -inf, l_old = 0
        # Online softmax tracks running max (m) and sum (l) across K/V chunks

        # Process K/V blocks - work in tile units, not pixels
        for kv_idx in range(NUM_KV_BLOCKS):
            # Pop Q, K, V blocks (Q repeated each iteration for now - inefficient but simpler)
            Q_block_mem = Q_cb.pop()
            K_block_mem = K_cb.pop()
            V_block_mem = V_cb.pop()

            # FIXME: Step 1 - Compute attention scores: S = Q @ K^T
            # Need matmul primitive (Q_block_mem @ K_block_mem.transpose())

            # FIXME: Step 2 - Scale by 1/sqrt(d_head)
            # Need: S_scaled = S / sqrt(d_head)
            # Requires: sqrt() function and division or multiplication by constant

            # FIXME: Step 3 - Apply softmax (online variant for memory efficiency)
            # Need primitives for numerically stable softmax:
            #   m_new = rowmax(S_scaled, m_old)  # Running max
            #   P = exp(S_scaled - m_new)         # Numerically stable exp
            #   l_new = exp(m_old - m_new) * l_old + rowsum(P)  # Running sum
            # This is the core FlashAttention optimization - softmax in blocks

            # FIXME: Step 4 - Compute attention output: O = softmax(S) @ V
            # Need: O_block = (P / l_new) @ V_block_mem
            # Then update running accumulator:
            #   O_acc = (l_old / l_new) * exp(m_old - m_new) * O_acc + O_block
            # This accumulates contributions from each K/V block

            # Reserve output BEFORE computation (to ensure memref is allocated)
            out_block_mem = out_cb.reserve()

            # Placeholder: simplest possible computation
            temp = Q_block_mem + K_block_mem

            # Store result
            out_block_mem.store(temp)
            out_cb.pop()

    @datamovement()
    async def dm_reader(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        cy = core_index(0)
        cx = core_index(1)
        q_idx = cy * GX + cx

        # Load Q, K, V in sync with compute kernel
        for kv_idx in range(NUM_KV_BLOCKS):
            # Load Q tile (inefficient - should only load once, but matches compute pattern)
            Q_shard = Q_cb.reserve()
            tx_q = dma(Q_stream[q_idx, 0], Q_shard)
            tx_q.wait()

            # Load K tile
            K_shard = K_cb.reserve()
            tx_k = dma(K_stream[kv_idx, 0], K_shard)
            tx_k.wait()

            # Load V tile
            V_shard = V_cb.reserve()
            tx_v = dma(V_stream[kv_idx, 0], V_shard)
            tx_v.wait()

    # Assemble program (no writer - output handled automatically)
    return Program(attention_compute, dm_reader)(Q, K, V, out)


# Simple test case
if __name__ == "__main__":
    # Start with 2D tensors (framework doesn't handle 4D yet)
    # Use exactly 1 tile (32x32) to eliminate multi-tile complexity
    tile_size = 32
    seq_len = tile_size
    d_head = tile_size

    Q = torch.randn(seq_len, d_head)
    K = torch.randn(seq_len, d_head)
    V = torch.randn(seq_len, d_head)
    out = torch.zeros(seq_len, d_head)

    flash_attention_simple(Q, K, V, out)

    # For now, just check it runs (output will be wrong)
    print("FlashAttention infrastructure test passed!")
    print(f"Output shape: {out.shape}")
    print(f"Output sample: {out[:4, :4]}")
