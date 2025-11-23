"""
Script to check GPU usage during training.
"""
import torch
import sys

def check_gpu_setup():
    """Checks GPU and PyTorch setup."""
    print("=" * 60)
    print("CHECKING GPU SETUP")
    print("=" * 60)
    
    # Check PyTorch
    print(f"\n1. PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"2. CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA NOT AVAILABLE!")
        print("   Possible reasons:")
        print("   - CPU version of PyTorch is installed")
        print("   - NVIDIA drivers are not installed")
        print("   - CUDA toolkit is not installed")
        print("\n   Solution:")
        print("   Install CUDA version of PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # GPU Information
    print(f"\n3. Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {props.name}")
        print(f"   - Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - Compute Capability: {props.major}.{props.minor}")
        print(f"   - Multiprocessors: {props.multi_processor_count}")
    
    # Check current device
    current_device = torch.cuda.current_device()
    print(f"\n4. Current device: GPU {current_device}")
    
    # GPU test
    print("\n5. GPU working test...")
    try:
        # Create test tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✓ GPU is working correctly!")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   - Memory allocated: {memory_allocated:.2f} GB")
        print(f"   - Memory reserved: {memory_reserved:.2f} GB")
        
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Error working with GPU: {e}")
        return False
    
    # Check CUDA version
    if hasattr(torch.version, 'cuda'):
        print(f"\n6. CUDA Version (PyTorch): {torch.version.cuda}")
    
    print("\n" + "=" * 60)
    print("✓ ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


def check_training_setup():
    """Checks training settings."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR TRAINING OPTIMIZATION")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("\n⚠ GPU not available. Training will be very slow on CPU.")
        return
    
    props = torch.cuda.get_device_properties(0)
    total_memory_gb = props.total_memory / 1024**3
    
    print(f"\nYour GPU: {props.name}")
    print(f"Available memory: {total_memory_gb:.2f} GB")
    
    # Recommendations for batch_size
    if total_memory_gb >= 8:
        recommended_batch = 64
        print(f"\n✓ Recommended batch_size: {recommended_batch}")
    elif total_memory_gb >= 6:
        recommended_batch = 32
        print(f"\n✓ Recommended batch_size: {recommended_batch}")
    else:
        recommended_batch = 16
        print(f"\n⚠ Recommended batch_size: {recommended_batch} (low memory)")
    
    print("\nOptimizations in train_im2latex.py:")
    print("  ✓ pin_memory=True - acceleration of transfer to GPU")
    print("  ✓ prefetch_factor=2 - batch prefetching")
    print("  ✓ non_blocking=True - asynchronous transfer")
    print("  ✓ Increased batch_size for better GPU utilization")
    
    print("\n⚠ IMPORTANT:")
    print("  - If GPU is not utilized, the bottleneck is data loading")
    print("  - With num_workers=0 on Windows loading is synchronous")
    print("  - GPU idles while images are loading")
    print("  - This is normal for Windows - GPU will not be utilized 100%")
    print("  - But training should be faster with optimizations")


if __name__ == "__main__":
    success = check_gpu_setup()
    if success:
        check_training_setup()
    else:
        print("\n⚠ Install CUDA version of PyTorch to use GPU!")
        sys.exit(1)

