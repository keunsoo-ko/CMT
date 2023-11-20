import torch
import numpy as np

def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def load_checkpoint(path, model, optimizer=None, reset_optimizer=True, is_dis=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    if is_dis:
        s = checkpoint["disc"]
    else:
        s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=True)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    return model

def psnr(img1, img2):
    mse = np.mean((img1-img2)** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

