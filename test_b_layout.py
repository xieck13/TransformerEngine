#!/usr/bin/env python3

"""Test B matrix layout for 1D1D kernel"""

import torch
import deep_gemm

device = torch.device('cuda')

# Create test data
A_fp32 = torch.randn(128, 128, device=device, dtype=torch.float32)
B_fp32 = torch.randn(128, 128, device=device, dtype=torch.float32)

# Cast A with per_token (rowwise)
A_data, A_scales = deep_gemm.per_token_cast_to_fp8(A_fp32, torch.float8_e4m3fn)
print(f"A_data: {A_data.shape}, A_scales: {A_scales.shape}")

# Test different B matrix layouts and quantization
test_cases = [
    ("B as-is, per_token", B_fp32, deep_gemm.per_token_cast_to_fp8),
    ("B transposed, per_token", B_fp32.t(), deep_gemm.per_token_cast_to_fp8),
    ("B as-is, per_block", B_fp32, deep_gemm.per_block_cast_to_fp8),
    ("B transposed, per_block", B_fp32.t(), deep_gemm.per_block_cast_to_fp8),
]

for name, B_matrix, cast_fn in test_cases:
    try:
        B_data, B_scales = cast_fn(B_matrix, torch.float8_e4m3fn)
        print(f"\n{name}: B_data={B_data.shape}, B_scales={B_scales.shape}")

        output = torch.empty(128, 128, device=device, dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt(
            (A_data, A_scales),
            (B_data, B_scales),
            output
        )
        print(f"  ✓ Success with NT layout")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Also test with NN layout
    try:
        output = torch.empty(128, 128, device=device, dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nn(
            (A_data, A_scales),
            (B_data, B_scales),
            output
        )
        print(f"  ✓ Success with NN layout")
    except Exception as e:
        print(f"  ✗ NN layout failed: {e}")