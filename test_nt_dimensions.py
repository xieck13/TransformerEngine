#!/usr/bin/env python3

"""Test different tensor dimensions to find what works for DeepGEMM NT layout"""

import torch
import deep_gemm

device = torch.device('cuda')

# Test different dimension combinations for NT layout
test_cases = [
    # (m, k, n) - A: [m, k], B: [n, k], C: [m, n]
    (128, 128, 128),  # Perfect squares
    (128, 256, 128),  # Our failing case dimensions
    (256, 128, 256),  # Swapped version
    (128, 128, 256),  # Different n
    (256, 256, 256),  # Larger squares
    (128, 128, 512),  # Even larger n
    (512, 128, 128),  # Larger m
]

print("Testing DeepGEMM NT layout with different dimensions:")
print("Format: A[m,k] @ B.T[k,n] -> C[m,n] where B is [n,k]")
print()

for m, k, n in test_cases:
    try:
        print(f"Testing dimensions: A[{m},{k}] @ B.T[{k},{n}] -> C[{m},{n}]")

        # Create test tensors
        A_fp32 = torch.randn(m, k, device=device, dtype=torch.float32)
        B_fp32 = torch.randn(n, k, device=device, dtype=torch.float32)  # B is [n, k] for NT

        # Quantize
        A_data, A_scales = deep_gemm.per_token_cast_to_fp8(A_fp32, torch.float8_e4m3fn)
        B_data, B_scales = deep_gemm.per_block_cast_to_fp8(B_fp32, torch.float8_e4m3fn)

        print(f"  A_data: {A_data.shape}, A_scales: {A_scales.shape}")
        print(f"  B_data: {B_data.shape}, B_scales: {B_scales.shape}")

        # Create output
        output = torch.empty(m, n, device=device, dtype=torch.bfloat16)

        # Try NT GEMM
        deep_gemm.fp8_gemm_nt(
            (A_data, A_scales),
            (B_data, B_scales),
            output,
            c=None,
            recipe=None
        )
        print(f"  ✅ Success! Output: {output.shape}")

        # Quick correctness check
        reference = torch.matmul(A_fp32, B_fp32.t())
        diff = torch.abs(output.float() - reference).max().item()
        print(f"  ✅ Max difference: {diff:.6f}")
        print()

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print()