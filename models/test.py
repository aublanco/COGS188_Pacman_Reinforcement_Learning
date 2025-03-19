import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return >=1 if a GPU is available
print(torch.cuda.get_device_name(0))  # Should print your NVIDIA GPU name