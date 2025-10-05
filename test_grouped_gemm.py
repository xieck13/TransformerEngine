#!/usr/bin/env python3

"""Test grouped GEMM functions directly"""

import torch
import deep_gemm

device = torch.device('cuda')

# Create test data for grouped GEMM
M, K, N = 256, 128, 128
A_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
B_fp32 = torch.randn(K, N, device=device, dtype=torch.float32)

# Cast to FP8
A_data, A_scales = deep_gemm.per_token_cast_to_fp8(A_fp32, torch.float8_e4m3fn)
B_data, B_scales = deep_gemm.per_block_cast_to_fp8(B_fp32, torch.float8_e4m3fn)

print(f"A_data: {A_data.shape}, A_scales: {A_scales.shape}")
print(f"B_data: {B_data.shape}, B_scales: {B_scales.shape}")

# Test m_splits
m_splits = torch.tensor([128, 128], dtype=torch.int, device=device)  # Use tensor, not list

output = torch.empty(M, N, device=device, dtype=torch.bfloat16)

print(f"Testing grouped GEMM with m_splits={m_splits}")

# Test NT layout
try:
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (A_data, A_scales),
        (B_data, B_scales),
        output,
        m_splits,
        recipe=None
    )
    print("✓ m_grouped_fp8_gemm_nt_contiguous works")
except Exception as e:
    print(f"✗ m_grouped_fp8_gemm_nt_contiguous failed: {e}")

# Test NN layout
try:
    output_nn = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nn_contiguous(
        (A_data, A_scales),
        (B_data, B_scales),
        output_nn,
        m_splits,
        recipe=None
    )
    print("✓ m_grouped_fp8_gemm_nn_contiguous works")
except Exception as e:
    print(f"✗ m_grouped_fp8_gemm_nn_contiguous failed: {e}")