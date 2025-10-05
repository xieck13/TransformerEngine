#!/usr/bin/env python3

"""Quick LinearDeepGemm accuracy test for comparison"""

import torch
from transformer_engine.pytorch.module.linear_deepgemm import LinearDeepGemm

device = 'cuda'
batch_size = 256
in_features = 256
out_features = 512

torch.manual_seed(42)
input_tensor = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)

# LinearDeepGemm
linear_deepgemm = LinearDeepGemm(
    in_features=in_features,
    out_features=out_features,
    bias=True,
    accumulate_into_main_grad=False,
    device=device,
    dtype=torch.bfloat16
)

# Reference
linear_ref = torch.nn.Linear(in_features, out_features, device=device, dtype=torch.bfloat16)
with torch.no_grad():
    linear_ref.weight.copy_(linear_deepgemm.weight)
    linear_ref.bias.copy_(linear_deepgemm.bias)

# Forward/backward
output_deepgemm = linear_deepgemm(input_tensor.clone())
loss_deepgemm = output_deepgemm.sum()
loss_deepgemm.backward()

input_ref = input_tensor.clone().detach().requires_grad_(True)
output_ref = linear_ref(input_ref)
loss_ref = output_ref.sum()
loss_ref.backward()

# Compare gradients
if linear_deepgemm.weight.grad is not None and linear_ref.weight.grad is not None:
    abs_error = torch.abs(linear_deepgemm.weight.grad - linear_ref.weight.grad)
    max_abs_error = abs_error.max().item()
    eps = 1e-8
    rel_error = abs_error / (torch.abs(linear_ref.weight.grad) + eps)
    max_rel_error = rel_error.max().item()

    print(f"LinearDeepGemm vs Reference:")
    print(f"  Max absolute error: {max_abs_error:.8f}")
    print(f"  Max relative error: {max_rel_error:.6f}")