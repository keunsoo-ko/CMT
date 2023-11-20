from network.swin import *
import torch.nn as nn
import torch

class Refine(nn.Module):
    def __init__(self, in_c):
        super(Refine, self).__init__()
        dim = 32

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c+1, dim // 2, kernel_size=7, stride=1, padding=3),  nn.GELU()), # 256, 256
            PatchEmbed(img_size=(256, 256), patch_size=4, in_chans=dim // 2, embed_dim=dim, norm_layer=nn.LayerNorm), # 64, 64, c
            BasicLayer(dim=dim, input_resolution=(64, 64), num_heads=1, depth=2, stride=2, window_size=4), # 32, 32, c * 2

            BasicLayer(dim=dim * 2, input_resolution=(32, 32), num_heads=2, depth=2, stride=2, window_size=4), # 16, 16, c * 4
            # 12 , 12
            BasicLayer(dim=dim * 4, input_resolution=(16, 16), num_heads=4, depth=2, stride=2, window_size=4), # 8, 8, c * 8
            # 6, 6
            BasicLayer(dim=dim * 8, input_resolution=(8, 8), num_heads=8, depth=4, stride=2, window_size=4),  # 4, 4, c * 16
            nn.Sequential(
                BasicLayer(dim=dim * 16, input_resolution=(4, 4), num_heads=16, depth=2, stride=None, window_size=2),
                AvgPool(dim * 16), nn.Linear(dim * 16, dim * 16), nn.ReLU())])  # 1, 1


        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 16, dim * 16), nn.ReLU(), UpSample((1, 1), dim * 16, dim * 16, 4)),  # 4, 4
            nn.Sequential(
                BasicLayer(dim=dim * 32, input_resolution=(4, 4), num_heads=16, depth=2, stride=None, window_size=2),
                UpSample((4, 4), dim * 32, dim * 16, 2)),  # 8, 8
            nn.Sequential(
                BasicLayer(dim=dim * 16 + dim * 8, input_resolution=(8, 8), num_heads=8, depth=4, stride=None, window_size=4),
                UpSample((8, 8), dim * 16 + dim * 8, dim * 8, 2)),  # 16, 16
            nn.Sequential(
                BasicLayer(dim=dim * 8 + dim * 4, input_resolution=(16, 16), num_heads=4, depth=2, stride=None, window_size=4),
                UpSample((16, 16), dim * 8 + dim * 4, dim * 4, 2)),  # 32, 32
            nn.Sequential(
                BasicLayer(dim=dim * 4 + dim * 2, input_resolution=(32, 32), num_heads=2, depth=2, stride=None, window_size=4),
                UpSample((32, 32), dim * 4 + dim * 2, dim, 2)), # 64, 64
            nn.Sequential(
                BasicLayer(dim=2 * dim, input_resolution=(64, 64), num_heads=1, depth=2, stride=None, window_size=4),
                UpSample((64, 64), 2 * dim, dim // 2, 4)),  # 256, 256
            nn.Sequential(
                BasicLayer(dim=dim, input_resolution=(256, 256), num_heads=1, depth=2, stride=None, window_size=8),
                Mlp(dim, out_features=3), nn.Tanh()),  # 96, 96
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        feats = []
        for f in self.encoder_blocks:
            x = f(x)
            feats.append(x) # [256,64,32,16,8,4,1]
        feats.pop()
        for i, f in enumerate(self.decoder_blocks):
            if i == 0:
                x = f(x)
            else:
                feat = feats[-1]
                if len(feat.shape) > 3:
                    feat = feat.view(feat.size(0), feat.size(1), -1).permute(0, 2, 1)
                x = f(torch.cat((x, feat), -1))
                feats.pop()

        outputs = x.reshape(B, H, W, 3).permute(0, 3, 1, 2)


        return outputs
