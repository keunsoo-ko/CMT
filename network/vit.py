import cv2
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Window_partition(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = pair(window_size)
    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.window_size
        sH, sW = kH // 2, kW // 2
        y = x.unfold(2, kH, kH).unfold(3, kW, kW) # B, C, nH, nW, kH, kW
        windows = y.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, kH*kW*C)

        y_ = x[..., sH:-sH, sW:-sW].unfold(2, kH, kH).unfold(3, kW, kW) # B, C, nH, nW, kH, kW
        windows_ = y_.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, kH*kW*C)

        return torch.cat([windows, windows_], 1)

def window_reverse(windows, window_size, resolution):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (list): window size (height, width)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    H, W = pair(resolution)
    kH, kW = pair(window_size)
    sH, sW = kH // 2, kW // 2
    num = (H//kH) * (W//kW)
    B = windows.shape[0]
    x = windows[:, :num].view(B, H//kH, W//kW, kH, kW, 3)
    y = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, H, W)

    y_ = y.clone()
    x_ = windows[:, num:].view(B, H//kH-1, W//kW-1, kH, kW, 3)
    y_[..., sH:-sH, sW:-sW] = x_.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, H-kH, W-kW)

    return [y, y_]



# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask=None):
        if mask is None:
            return self.fn(self.norm(x))
        else:
            return self.fn(self.norm(x), mask)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., window=16, resolution=256, is_overlap=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.window = window
        self.resolution = resolution
        self.is_overlap = is_overlap
        self.pooling = nn.AvgPool2d(2, stride=1)

    def forward(self, x, mask):
        B = x.shape[0]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        with torch.no_grad():
            updated = (torch.mean(mask[:, 1:], dim=-1, keepdim=True) > 0.) * 1.
            updated = torch.cat([torch.ones_like(updated[:, :1]), updated], 1)
            qkv_m = mask @ torch.abs(self.to_qkv.weight.transpose(1, 0))
            qkv_m = qkv_m.chunk(3, dim = -1)
            q_m, k_m, v_m = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_m)
            q_m = q_m / (1e-6 + torch.max(q_m, dim=2, keepdim=True)[0])
            k_m = k_m / (1e-6 + torch.max(k_m, dim=2, keepdim=True)[0])
            v_m = v_m / (1e-6 + torch.max(v_m, dim=2, keepdim=True)[0])

        dots = torch.matmul(q * q_m, (k * k_m).transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v_m * v)
        out = rearrange(out, 'b h n d -> b n (h d)') * updated 

        with torch.no_grad():
            m = torch.matmul(attn, v_m)
            m = rearrange(m, 'b h n d -> b n (h d)') * updated


        # bridge update for bridge tokens
        kH, kW = pair(self.window)
        H, W = pair(self.resolution)
        nH, nW = H // kH, W // kW

        if self.is_overlap:
            ## update region for bridge
            out_bridge = out[:, 1:nH*nW+1].reshape(B, nH, nW, -1).unfold(1, 2, 1).unfold(2, 2, 1) # B, nH-1, nW-1, C, 2, 2
            out_bridge = torch.mean(out_bridge, (-2, -1))
            with torch.no_grad():
                m_bridge = m[:, 1:nH*nW+1].reshape(B, nH, nW, -1).unfold(1, 2, 1).unfold(2, 2, 1) # B, nH-1, nW-1, C, 2, 2
                m_bridge = torch.mean(m_bridge, (-2, -1))
                # average & update
                m_bridge = m_bridge.reshape(B, (nH-1) * (nW-1), -1) * (1 - updated[:, -(nH-1)*(nW-1):])

            # average & update 
            out_bridge = out_bridge.reshape(B, (nH-1) * (nW-1), -1) * (1 - updated[:, -(nH-1)*(nW-1):])


            ## update region for origin
            out_origin = F.pad(out[:, -(nH-1)*(nW-1):].reshape(B, nH-1, nW-1, -1), (0, 0, 1, 1, 1, 1), value=0)  
            out_origin = out_origin.unfold(1, 2, 1).unfold(2, 2, 1) # B, nH, nW, C, 2, 2
            out_origin = torch.mean(out_origin, (-2, -1))
            with torch.no_grad():
                m_origin = F.pad(m[:, -(nH-1)*(nW-1):].reshape(B, nH-1, nW-1, -1), (0, 0, 1, 1, 1, 1), value=0)
                m_origin = m_origin.unfold(1, 2, 1).unfold(2, 2, 1) # B, nH, nW, C, 2, 2
                m_origin = torch.mean(m_origin, (-2, -1))

                # average & update
                m_origin = m_origin.reshape(B, nH*nW, -1) * (1 - updated[:, 1:nH*nW+1])

                # final update
                m[:, 1:nH*nW+1] += m_origin
                m[:, -(nH-1)*(nW-1):] += m_bridge

            # average & update
            out_origin = out_origin.reshape(B, nH*nW, -1) * (1 - updated[:, 1:nH*nW+1])

            # final update
            out[:, 1:nH*nW+1] += out_origin
            out[:, -(nH-1)*(nW-1):] += out_bridge


        with torch.no_grad():
            inter = F.interpolate(torch.mean(m[:, 1:], dim=-1, keepdim=True)[:, :16*16].reshape(-1, 1, 16, 16), (256, 256))
            inter_shift = F.interpolate(torch.mean(m[:, 1:], dim=-1, keepdim=True)[:, -15*15:].reshape(-1, 1, 15, 15), (15*16, 15*16))
            inter[..., 8:-8, 8:-8] = (inter[..., 8:-8, 8:-8] + inter_shift) / 2
            m = m @ torch.abs(self.to_out[0].weight.transpose(1, 0))



        return self.to_out(out), m, inter

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., window=16, resolution=256):
        super().__init__()
        self.layers = nn.ModuleList([])
        is_overlap = True
        kH, kW = pair(window)
        H, W = pair(resolution)
        self.num = (H//kH)*(W//kW)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, window=window, resolution=resolution, is_overlap=is_overlap)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, m):
        stack = []
        for i, (attn, ff) in enumerate(self.layers):
            y, m, inter = attn(x, m)
            x = y + x
            x = ff(x) + x
            stack.append(inter)
        return x, stack

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_h_patches = image_height // patch_height
        num_w_patches = image_width // patch_width
        patch_dim = (channels + 1) * patch_height * patch_width
        out_dim = channels * patch_height * patch_width

        self.to_patch = nn.Sequential(
            Window_partition(patch_size),
            nn.Linear(patch_dim, dim),
        )
        num_patches = num_h_patches*num_w_patches + (num_h_patches-1)*(num_w_patches-1)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, patch_size, image_size)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
            nn.Tanh()
        )

        self.window = patch_size
        self.resolution = image_size

        #self.to_image = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = num_h_patches, w = num_w_patches, p1 = patch_height, p2 = patch_width)
        self.mask_to_flat = Window_partition(patch_size)

    def forward(self, img, mask):
        mask = 1 - mask
        x = self.to_patch(torch.cat((img, mask), 1))
        # mask updated
        with torch.no_grad():
            m = self.mask_to_flat(mask.repeat(1, 4, 1, 1)) # b, n, 1
            m = torch.cat((torch.ones_like(m[:, :1]), m), dim=1)
            m = m @ torch.abs(self.to_patch[1].weight.transpose(1, 0))

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        #inter_m = m / (1e-6 + torch.max(m, dim=2, keepdim=True)[0])
        #inter = F.interpolate(torch.mean(inter_m[:, 1:], dim=-1, keepdim=True)[:, :16*16].reshape(-1, 1, 16, 16), (256, 256))
        #inter_shift = F.interpolate(torch.mean(inter_m[:, 1:], dim=-1, keepdim=True)[:, -15*15:].reshape(-1, 1, 15, 15), (15*16, 15*16))
        #inter[..., 8:-8, 8:-8] = (inter[..., 8:-8, 8:-8] + inter_shift) / 2


        x, stack = self.transformer(x, m)
        x = x[:, 1:]
        x = self.mlp_head(x) # B, S, 3

        out = window_reverse(x, self.window, self.resolution)

        return out, stack

