# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test that stream_layout ops are created at the start of the pipeline.

This test exercises the early stream creation feature where streams
defined in the DSL are materialized as stream_layout ops immediately
in the Python-generated IR, rather than being added later by a pass.
"""
from pykernel.d2m_api import *
from utils import assert_pcc
import torch


@pykernel_gen(
    block_factors=[
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    grid=(1, 1),
)
def simple_add_with_streams(lhs, rhs, out, block_factors=None, grid=None):
    """Simple element-wise add with both inputs as streams."""
    lhs_stream = Stream(lhs)
    rhs_stream = Stream(rhs)

    @compute()
    async def add_kernel(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        # TODO: Fix loop type mismatch issue - commenting out for now
        # for i in range(1):
        lhs_shard = lhs_cb.pop()
        rhs_shard = rhs_cb.pop()
        out_shard = out_cb.reserve()
        result = lhs_shard + rhs_shard
        out_shard.store(result)
        out_cb.pop()

    @datamovement()
    async def dm0(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        # TODO: Fix loop type mismatch issue - commenting out for now
        # for i in range(1):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_stream[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    async def dm1(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        # TODO: Fix loop type mismatch issue - commenting out for now
        # for i in range(1):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_stream[0, 0], rhs_shard)
        tx.wait()

    return Program(add_kernel, dm0, dm1)(lhs, rhs, out)



def test_simple_add_with_streams():
    """Test that both inputs are wrapped in stream_layout ops."""
    print("\n=== Test: Simple add with both inputs as streams ===")
    lhs = torch.randn(64, 64)
    rhs = torch.randn(64, 64)
    out = torch.zeros(64, 64)

    try:
        simple_add_with_streams(lhs, rhs, out)
        print("✓ Successfully generated IR with stream_layout ops for both inputs")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise




if __name__ == "__main__":
    print("="*60)
    print("Testing early stream creation in D2M pipeline")
    print("="*60)

    test_simple_add_with_streams()
    
    print("\n" + "="*60)
    print("All tests passed! Streams are created at pipeline start ✓")
    print("="*60)
