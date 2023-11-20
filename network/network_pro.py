import torch.nn as nn
import torch
from network.vit import ViT
from network.refine import Refine


class Inpaint(nn.Module):
    def __init__(self, input_size=256, patch_size=16, depth=15, heads=16):
        super().__init__()
        self.coarse = ViT(input_size, patch_size, (patch_size**2) * 3, depth, heads, 1024)
        self.refine = Refine(6)

    def forward(self, img, mask):
        c_gen, stack = self.coarse(img * (1 - mask), mask)
        c_gen_ = []
        for c_g in c_gen:
            c_gen_.append((c_g * mask) + img * (1 - mask))
        gen = self.refine(torch.cat(c_gen_+[mask], 1))
        gen = (gen * mask) + img * (1 - mask)
        return gen
