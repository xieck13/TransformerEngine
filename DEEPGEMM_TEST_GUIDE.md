# DeepGEMM Test Guide

This guide provides step-by-step instructions for testing the FP8DeepGemm integration with TransformerEngine in a GPU environment.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with SM90 (H100) or SM100 (H200/B200) architecture for optimal performance
- Any CUDA-capable GPU for basic functionality testing
- At least 8GB GPU memory recommended

### Software Requirements
- Python 3.8+
- PyTorch 2.1+
- CUDA Toolkit 12.3+ (12.9+ recommended)
- TransformerEngine
- DeepGEMM library (optional - graceful fallback if not available)

## Test Environment Setup

### Option 1: Docker Environment (Recommended)

```bash
# Use NVIDIA's PyTorch container with CUDA support
docker run --gpus all -it --rm \
  -v /path/to/TransformerEngine:/workspace/TransformerEngine \
  nvcr.io/nvidia/pytorch:24.02-py3

# Inside the container
cd /workspace/TransformerEngine
```

### Option 2: Native Environment

```bash
# Ensure CUDA is available
nvidia-smi

# Install dependencies if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformer-engine[pytorch]
```

## DeepGEMM Installation (Optional)

For maximum performance, install DeepGEMM:

```bash
# Clone and build DeepGEMM
cd /tmp
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
./install.sh

# Verify installation
python -c "import deep_gemm; print('DeepGEMM installed successfully')"
```

## Running Tests

### 1. Quick Validation Test

First, run a quick test to verify the basic setup:

```bash
cd /workspace/TransformerEngine

# Quick import test
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')

try:
    import deep_gemm
    print('✓ DeepGEMM available')
except ImportError:
    print('! DeepGEMM not available (will use fallback)')

# Test basic TransformerEngine import
import transformer_engine_torch as tex
print('✓ TransformerEngine imported successfully')
"
```

### 2. Comprehensive Integration Test

Run the main test suite:

```bash
python test_fp8_deepgemm_integration.py
```

Expected output structure:
```
Comprehensive FP8DeepGemmQuantizer Integration Test
============================================================

=== Testing FP8DeepGemmQuantizer Creation ===
✓ Created FP8DeepGemmQuantizer (DeepGEMM available: True/False)
  - FP8 dtype: ...
  - Block size: ...
  - Use DeepGEMM layout: True

=== Testing Tensor Quantization ===
✓ Created test tensor: torch.Size([256, 512]) on cuda:0
✓ Created empty quantized tensor: FP8DeepGemmQTensor
✓ Successfully quantized tensor
✓ Dequantized tensor shape: torch.Size([256, 512])
✓ Quantization error (MAE): 0.001234

=== Testing DeepGEMM GEMM Operations ===
✓ Created quantized tensors: A(256, 512), B(1024, 512)
✓ DeepGEMM GEMM result shape: torch.Size([256, 1024])
✓ Regular GEMM result shape: torch.Size([256, 1024])

=== Testing LinearDeepGemm Module ===
✓ Created LinearDeepGemm module: ...
✓ Created input tensor: torch.Size([128, 512])
✓ Forward pass successful: torch.Size([128, 512]) -> torch.Size([128, 1024])
✓ Output dtype: torch.bfloat16
✓ Backward pass successful

=== Testing LayerNormLinearDeepGemm Module ===
✓ Created LayerNormLinearDeepGemm module: ...
✓ Forward pass successful: torch.Size([128, 512]) -> torch.Size([128, 1024])
✓ Backward pass successful

=== Testing GroupedLinearDeepGemm Module ===
✓ Created GroupedLinearDeepGemm module: ...
✓ Using m_splits: [32, 32, 32, 32]
✓ Forward pass successful: torch.Size([128, 512]) -> torch.Size([128, 512])
✓ Backward pass successful

============================================================
✓ All tests completed with DeepGEMM available
🎉 All tests passed successfully!
```

### 3. Performance Benchmark Test

Create and run a performance test:

```bash
cat > performance_test.py << 'EOF'
#!/usr/bin/env python3

import time
import torch
import sys
sys.path.insert(0, '/workspace/TransformerEngine')

from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer, DEEPGEMM_AVAILABLE
from transformer_engine.pytorch.module.linear_deepgemm import LinearDeepGemm
from transformer_engine_torch import DType as TE_DType

def benchmark_linear_layers():
    print("Performance Benchmark: Linear Layers")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"DeepGEMM Available: {DEEPGEMM_AVAILABLE}")

    # Test configuration
    batch_size = 128
    seq_len = 2048
    hidden_size = 4096
    num_iterations = 10

    # Create models
    regular_linear = torch.nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.bfloat16)
    deepgemm_linear = LinearDeepGemm(
        in_features=hidden_size,
        out_features=hidden_size,
        use_deepgemm=True,
        device=device,
        dtype=torch.bfloat16
    )

    # Create input
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        _ = regular_linear(input_tensor)
        _ = deepgemm_linear(input_tensor)

    torch.cuda.synchronize()

    # Benchmark regular linear
    start_time = time.time()
    for _ in range(num_iterations):
        output = regular_linear(input_tensor)
    torch.cuda.synchronize()
    regular_time = time.time() - start_time

    # Benchmark DeepGEMM linear
    start_time = time.time()
    for _ in range(num_iterations):
        output = deepgemm_linear(input_tensor)
    torch.cuda.synchronize()
    deepgemm_time = time.time() - start_time

    print(f"\nResults ({num_iterations} iterations):")
    print(f"Regular Linear:   {regular_time:.4f}s ({regular_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"DeepGEMM Linear:  {deepgemm_time:.4f}s ({deepgemm_time/num_iterations*1000:.2f}ms per iteration)")

    if deepgemm_time < regular_time:
        speedup = regular_time / deepgemm_time
        print(f"🚀 DeepGEMM is {speedup:.2f}x faster!")
    else:
        slowdown = deepgemm_time / regular_time
        print(f"⚠️  DeepGEMM is {slowdown:.2f}x slower (may be expected without proper GPU arch)")

if __name__ == "__main__":
    benchmark_linear_layers()
EOF

python performance_test.py
```

### 4. Memory Usage Test

Test memory efficiency:

```bash
cat > memory_test.py << 'EOF'
#!/usr/bin/env python3

import torch
import sys
sys.path.insert(0, '/workspace/TransformerEngine')

from transformer_engine.pytorch.module.linear_deepgemm import LinearDeepGemm
from transformer_engine_torch import DType as TE_DType

def test_memory_usage():
    print("Memory Usage Test")
    print("=" * 30)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {initial_memory / 1024**2:.2f} MB")

        # Create DeepGEMM linear layer
        linear = LinearDeepGemm(
            in_features=4096,
            out_features=4096,
            use_deepgemm=True,
            device=device,
            dtype=torch.bfloat16
        )

        model_memory = torch.cuda.memory_allocated() - initial_memory
        print(f"Model memory: {model_memory / 1024**2:.2f} MB")

        # Test forward pass
        input_tensor = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)
        output = linear(input_tensor)

        total_memory = torch.cuda.memory_allocated() - initial_memory
        print(f"Total memory after forward: {total_memory / 1024**2:.2f} MB")

        # Test backward pass
        loss = output.sum()
        loss.backward()

        final_memory = torch.cuda.memory_allocated() - initial_memory
        print(f"Final memory after backward: {final_memory / 1024**2:.2f} MB")

        print("✓ Memory test completed successfully")
    else:
        print("CPU mode - memory tracking not available")

if __name__ == "__main__":
    test_memory_usage()
EOF

python memory_test.py
```

## Test Scenarios and Expected Results

### Scenario 1: Full DeepGEMM Environment (H100/H200)
- **Expected**: All tests pass with optimal performance
- **DeepGEMM Available**: True
- **Performance**: Up to 1550 TFLOPS, significant speedup over regular GEMM

### Scenario 2: CUDA GPU without DeepGEMM
- **Expected**: All tests pass with fallback behavior
- **DeepGEMM Available**: False
- **Performance**: Similar to regular operations, but with quantization benefits

### Scenario 3: CPU-only Environment
- **Expected**: All tests pass but with warnings
- **Performance**: Slower but functional for development/testing

## Troubleshooting

### Common Issues and Solutions

1. **ImportError: No module named 'deep_gemm'**
   ```bash
   # This is expected - DeepGEMM will fallback gracefully
   # Look for "! DeepGEMM not available (will use fallback)" in test output
   ```

2. **CUDA out of memory**
   ```bash
   # Reduce batch sizes in tests
   export CUDA_VISIBLE_DEVICES=0  # Use single GPU
   # Or modify test parameters in test files
   ```

3. **Quantization errors**
   ```bash
   # Ensure tensor dimensions are multiples of block size (128)
   # The test automatically pads tensors to correct sizes
   ```

4. **Performance not improved**
   ```bash
   # Check GPU architecture
   nvidia-smi --query-gpu=name --format=csv
   # DeepGEMM requires SM90+ for optimal performance
   ```

### Debug Mode

Run tests with debug information:

```bash
# Enable verbose output
export DG_JIT_DEBUG=1
export DG_PRINT_CONFIGS=1

# Run with Python in debug mode
python -u test_fp8_deepgemm_integration.py 2>&1 | tee test_output.log
```

## Test Results Interpretation

### Success Indicators
- ✅ All imports successful
- ✅ Quantizer creation without errors
- ✅ Forward/backward passes complete
- ✅ Memory usage within expected ranges
- ✅ No CUDA errors or crashes

### Performance Indicators
- 🚀 Speedup over regular operations (with proper GPU)
- 📈 Memory efficiency (2x better than FP16)
- 🎯 Low quantization error (< 0.01 MAE)

### Warning Indicators (Expected)
- ⚠️  "DeepGEMM not available" - Normal without DeepGEMM installation
- ⚠️  "Non-FP8DeepGemmQTensor inputs" - Normal fallback behavior
- ⚠️  Performance similar to baseline - Expected without SM90+ GPU

## Next Steps

After successful testing:

1. **Integration**: Use the modules in your transformer models
2. **Optimization**: Fine-tune block scaling and quantization parameters
3. **Production**: Deploy with proper GPU architecture for maximum benefits

## Support

If you encounter issues:

1. Check the test output logs
2. Verify GPU architecture compatibility
3. Ensure all dependencies are correctly installed
4. Review the comprehensive error messages in test output

The implementation includes extensive error handling and fallback mechanisms, so most issues are environmental rather than code-related.