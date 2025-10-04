#!/usr/bin/env python3

"""Simple test to debug the exact tensor shape issue"""

import torch
import math

def test_scale_shapes():
    print("Testing Scale Shape Calculations")
    print("=" * 40)

    # Test the problematic dimensions
    test_cases = [
        (128, 128),   # Aligned
        (256, 512),   # Non-square aligned
        (512, 384),   # Partially aligned
    ]

    try:
        # Try to import DeepGEMM utilities
        from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

        device = torch.device('cuda')

        for M, K in test_cases:
            print(f"\nTesting tensor shape ({M}, {K})")

            # Create test tensor
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            print(f"Input shape: {x.shape}")

            # Test per_token_cast_to_fp8 (rowwise)
            try:
                data_token, scales_token = per_token_cast_to_fp8(x, use_ue8m0=False)
                print(f"per_token_cast_to_fp8:")
                print(f"  data: {data_token.shape}")
                print(f"  scales: {scales_token.shape}")

                # My calculation for rowwise
                my_rowwise_scale_shape = (M, math.ceil(K / 128))
                print(f"  my calculation: {my_rowwise_scale_shape}")
                print(f"  match: {scales_token.shape == my_rowwise_scale_shape}")

            except Exception as e:
                print(f"per_token_cast_to_fp8 failed: {e}")

            # Test per_block_cast_to_fp8 (columnwise)
            try:
                data_block, scales_block = per_block_cast_to_fp8(x, use_ue8m0=False)
                print(f"per_block_cast_to_fp8:")
                print(f"  data: {data_block.shape}")
                print(f"  scales: {scales_block.shape}")

                # My calculation for columnwise
                my_columnwise_scale_shape = (math.ceil(M / 128), math.ceil(K / 128))
                print(f"  my calculation: {my_columnwise_scale_shape}")
                print(f"  match: {scales_block.shape == my_columnwise_scale_shape}")

            except Exception as e:
                print(f"per_block_cast_to_fp8 failed: {e}")

    except ImportError:
        print("❌ Cannot import DeepGEMM utilities")

if __name__ == "__main__":
    test_scale_shapes()