import torch.nn.functional as F
import torch
from torch import nn

def Qcertain(Y, R ,N = 4):
    points_idx, points = sampling_points(Y, R, N=N, training=False)  # [B,N,坐标]

    # YU = point_sample(Y, points, align_corners=False)
    RU = point_sample(R, points, align_corners=False)

    B, C, H, W = Y.shape
    points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
    out = (Y.reshape(B, C, -1)
           .scatter_(2, points_idx, RU)
           .view(B, C, H, W))

    return out

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def sampling_points(Y, R, N, training=True):
    assert Y.dim() == 4, "Dim must be N(Batch)CHW"
    device = Y.device
    B, _, H, W = Y.shape
    # mask, _ = mask.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncer = torch.full([B, 1, H, W], 0.5, device=device)

        Yuncertainty_map = 0.5 - torch.abs((Y[:] - uncer))
        Rcertainty_map = torch.abs((R[:] - uncer))
        error_score = Yuncertainty_map*Rcertainty_map

        _, idx = error_score.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points

def Cal_uncertain(Y):
    assert Y.dim() == 4, "Dim must be N(Batch)CHW"
    device = Y.device
    B, C, H, W = Y.shape
    # mask, _ = mask.sort(1, descending=True)

    uncer = torch.full([B, 1, H, W], 0.5, device=device)
    Yuncertainty_map = torch.abs((Y[:] - uncer))

    unfold_ = nn.Unfold(kernel_size=(128, 128), dilation=1, padding=0, stride=128)
    Yuncertainty_map = unfold_(Yuncertainty_map).permute(0, 2, 1)

    # Yuncertainty_map[Yuncertainty_map >= 0.45] = 0#[b,4,hw]
    Yuncertainty = Yuncertainty_map.sum(2)
    # print(Yuncertainty)

    val, index = torch.topk(Yuncertainty, k=1, dim=1, largest=False)#求最小值索引

    return index.view(-1)

if __name__ == '__main__':
     Y = torch.randn([2, 1, 512, 512])
     # R = torch.randn([2, 1, 4, 4])
     # Y[Y < 0] = 0
     # Y[Y > 1] = 1
     # R[R < 0] = 0
     # R[R > 1] = 1
     # print(Y)
     # print(R)
     # print(Qcertain(Y,R))
     # Cal_uncertain(Y)
