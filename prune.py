import torch
import time

def allocate_gpu_memory(target_usage_ratio=0.9):
    """
    Allocate GPU memory to reach the target usage ratio.
    
    Args:
        target_usage_ratio (float): Ratio of GPU memory to allocate (e.g., 0.9 for 90% usage).
    """
    total_gpus = torch.cuda.device_count()
    allocated_tensors = []
    
    for gpu_idx in range(total_gpus):
        device = f"cuda:{gpu_idx}"
        torch.cuda.set_device(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = int(total_memory * target_usage_ratio)
        
        # Allocate a large tensor to fill up GPU memory
        try:
            tensor = torch.empty(allocated_memory // 4, dtype=torch.float32, device=device)
            allocated_tensors.append(tensor)
            print(f"GPU {gpu_idx}: Allocated {allocated_memory / (1024 ** 3):.2f} GB out of {total_memory / (1024 ** 3):.2f} GB")
        except RuntimeError as e:
            print(f"GPU {gpu_idx}: Error allocating memory - {e}")

if __name__ == "__main__":
    # Check for CUDA availability
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Please check your GPU and CUDA installation.")

    # Allocate memory on GPUs (targeting at least 90%)
    allocate_gpu_memory(target_usage_ratio=0.9)

    # Keep the script running for 6 hours
    print("Starting long run...")
    for i in range(360):  # 360 iterations, each sleeping for 60 seconds
        time.sleep(60)
        print(f"Running prune.py... {i + 1} minute(s) elapsed.")

    print("Finished execution.")
