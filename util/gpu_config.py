import torch

# use lightning tuner to deduce batch size for other models
def get_batch_size(config):
    batch_size = config['experiment']['batch_size']
    gpus = config['experiment']['gpus']
    name = torch.cuda.get_device_name(0)
    per_gpu = batch_size // gpus if batch_size is not None else float('inf')
    
    if '2080' in name:
        return min(1, per_gpu)
    elif '3090' in name or \
        ('TITAN' in name and 'RTX' in name) or \
        'V100' in name or \
        'A100' in name:
        return min(4, per_gpu)
    elif 'A40' in name:
        return min(8, per_gpu)
    else:
        return 1