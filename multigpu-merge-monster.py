# Example Device Setup for Multi-GPU Support

# Modified to support multi-GPU configurations
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    # Additional setup for DataParallel or DistributedDataParallel could go here

# Example Model Loading with DataParallel

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

# Example Tensor Operation Adjustments for Device

# Example tensor operation adjusted for device
tensor_a = tensor_a.to(device)
tensor_b = tensor_b.to(device)
result = tensor_a + tensor_b  # Example operation
