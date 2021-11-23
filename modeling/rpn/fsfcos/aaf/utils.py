import torch
import torch.nn as nn
import torch.nn.functional as F


class AAFIOMode:
    # mode for identity
    ID = 1
    # mode for query of the form BCHW and support NCHW
    Q_BCHW_S_NCHW = 2
    # mode for query of the form BNCHW and support NBCHW
    # note the permuted dimension between query and support
    Q_BNCHW_S_NBCHW = 3
    # mode for query of the form BNCHW
    Q_BNCHW = 5



def prime_factors(n):
    # Not efficient but n is always small
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


class BackgroundAttenuationBlock(nn.Module):
    """
    BackgroundAttenuationBlock for Dual-Awareness Attention for
    Few-Shot Object Detection
    (https://arxiv.org/abs/2102.12152) 
    """
    def __init__(self, in_features, n_level, cfg, *args):
        super().__init__(*args)
        self.cfg = cfg
        self.device = self.cfg.MODEL.DEVICE
        self.matrices = [nn.Linear(in_features, 1).to(self.device) for i in range(n_level)]

    def forward(self, features):
        out_features = []
        for feat, matrix in zip(features, self.matrices):
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w).permute(0, 2, 1) # B, HW, C
            attention = F.softmax(matrix(feat), dim=1) # B, HW, 1
            feat_weighted = torch.bmm(attention.permute(0,2,1), feat) # B, 1, C

            feat = feat + 0.1 * F.leaky_relu(feat_weighted)

            out_features.append(feat.permute(0,2,1).view(b, c, h, w))
        return out_features

class RelationGRU(nn.Module):
    """
    Relation GRU for Few-Shot Object Detection With Self-Adaptive
    Attention Network for Remote Sensing Images
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9426416) 
    """
    def __init__(self, d):
        super(RelationGRU, self).__init__()
        self.d = d

        self.Wr = nn.Linear(2 * d, d)
        self.Wz = nn.Linear(2 * d, d)
        self.Wn = nn.Linear(d, d)
        self.Un = nn.Linear(d, d)

    def forward(self, x_seq, h):
        # x_seq: L, B, d
        # h: B, d
        n_seq = []
        for x in x_seq:
            r = torch.sigmoid(self.Wr(torch.cat([x, h], dim=-1)))
            z = torch.sigmoid(self.Wz(torch.cat([x, h], dim=-1)))
            n_t = torch.tanh(self.Wn(x) + self.Un(r * h))
            ht = z * h + (1 - z) * n_t
            n_seq.append(n_t)
        return torch.stack(n_seq)  # L, B, d
