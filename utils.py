import torch

def perm_generator(length):
    seen = set()
    while True:
        perm = tuple(torch.randperm(length).tolist())
        if perm not in seen:
            seen.add(perm)
            yield perm


