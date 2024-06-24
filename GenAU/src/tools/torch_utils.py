import torch

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def random_uniform(start, end):
    val = torch.rand(1).item()
    return start + (end - start) * val

def print_on_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)