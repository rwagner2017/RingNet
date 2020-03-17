import torch
num_devices = torch.cuda.device_count()
if num_devices > 0:
    device_id = torch.cuda.current_device()
    print(torch.cuda.device(device_id))
    print(torch.cuda.get_device_name(device_id))
else:
    print('no gpu-device attached')