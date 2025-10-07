#!/usr/bin/env python3

"""Test NN layout specifically to understand the constraint"""

import torch
import deep_gemm

device = torch.device('cuda')

# Create test data
A_fp32 = torch.randn(128, 128, device=device, dtype=torch.float32)
B_fp32 = torch.randn(128, 128, device=device, dtype=torch.float32)

# Cast with different approaches
A_data, A_scales = deep_gemm.per_token_cast_to_fp8(A_fp32, torch.float8_e4m3fn)
B_data, B_scales = deep_gemm.per_block_cast_to_fp8(B_fp32, torch.float8_e4m3fn)

print(f"A_data: {A_data.shape}, A_scales: {A_scales.shape}")
print(f"B_data: {B_data.shape}, B_scales: {B_scales.shape}")

# Test different B layouts for NN
test_cases = [
    ("B as-is", B_fp32),
    ("B transposed", B_fp32.t()),
    ("B contiguous", B_fp32.contiguous()),
    ("B transposed contiguous", B_fp32.t().contiguous()),
]

for name, B_matrix in test_cases:
    try:
        B_test_data, B_test_scales = deep_gemm.per_block_cast_to_fp8(B_matrix, torch.float8_e4m3fn)
        print(f"\n{name}: B_data={B_test_data.shape}, B_scales={B_test_scales.shape}")
        print(f"  B matrix is_contiguous: {B_matrix.is_contiguous()}")
        print(f"  B_test_data is_contiguous: {B_test_data.is_contiguous()}")

        output = torch.empty(128, 128, device=device, dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nn(
            (A_data, A_scales),
            (B_test_data, B_test_scales),
            output,
            c=None,
            recipe=None
        )
        print(f"  ✓ Success with NN layout")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Check if the issue is specific to fp8_gemm_nn vs other layouts
print(f"\n=== Testing NT vs NN with same data ===")
output_nt = torch.empty(128, 128, device=device, dtype=torch.bfloat16)
output_nn = torch.empty(128, 128, device=device, dtype=torch.bfloat16)

try:
    deep_gemm.fp8_gemm_nt((A_data, A_scales), (B_data, B_scales), output_nt)
    print("✓ NT layout works")
except Exception as e:
    print(f"✗ NT layout failed: {e}")

try:
    deep_gemm.fp8_gemm_nn((A_data, A_scales), (B_data, B_scales), output_nn)
    print("✓ NN layout works")
except Exception as e:
    print(f"✗ NN layout failed: {e}")