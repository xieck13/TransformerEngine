#!/usr/bin/env python3

"""Simple script to check TransformerEngine installation status"""

import sys
import os

def check_installation():
    print("TransformerEngine Installation Check")
    print("=" * 40)

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False

    # Check TransformerEngine compiled extension
    try:
        import transformer_engine.pytorch  # This loads the compiled extension
        print("✓ transformer_engine.pytorch loaded")

        import transformer_engine_torch as tex
        print(f"✓ transformer_engine_torch extension available")
        print(f"✓ TE version: {getattr(tex, '__version__', 'unknown')}")

        # Test basic functionality
        try:
            from transformer_engine_torch import DType as TE_DType
            dtype = TE_DType.kFloat8E4M3
            print(f"✓ FP8 dtype available: {dtype}")
        except Exception as e:
            print(f"! FP8 dtype issue: {e}")

    except ImportError as e:
        print(f"✗ TransformerEngine not properly installed: {e}")
        print("Options:")
        print("1. pip install transformer-engine[pytorch]")
        print("2. Build from source with CUDA support")
        return False

    # Check TransformerEngine Python modules
    try:
        import transformer_engine.pytorch
        print("✓ transformer_engine.pytorch available")

        from transformer_engine.pytorch.module import Linear
        print("✓ Basic modules available")

    except ImportError as e:
        print(f"✗ TransformerEngine Python modules not available: {e}")
        return False

    # Check if we can import our DeepGEMM classes
    try:
        from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer
        print("✓ FP8DeepGemmQuantizer available")

        from transformer_engine.pytorch.module.linear_deepgemm import LinearDeepGemm
        print("✓ LinearDeepGemm available")

    except ImportError as e:
        print(f"✗ DeepGEMM classes not available: {e}")
        print("Make sure you're running from the correct TransformerEngine directory")
        return False

    # Check DeepGEMM availability
    try:
        import deep_gemm
        print("✓ DeepGEMM library available")
    except ImportError:
        print("! DeepGEMM library not available (will use fallback)")

    print("\n" + "=" * 40)
    print("✅ TransformerEngine installation looks good!")
    return True

if __name__ == "__main__":
    success = check_installation()
    if not success:
        print("\n❌ Installation check failed. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n🎉 Ready to run DeepGEMM tests!")