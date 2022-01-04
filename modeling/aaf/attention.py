import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from .. import registry
from ...utils import apply_tensor_list
from .utils import BackgroundAttenuationBlock, AAFIOMode, RelationGRU


class BaseAttention(nn.Module):
    def __init__(self, cfg, align_first, *args):
        super().__init__(*args)

        self.cfg = cfg

        self.input_name = '_features' if not align_first else '_p1'
        self.output_name = '_p1' if not align_first else '_p2'

        self.in_mode = AAFIOMode.Q_BNCHW_S_NBCHW
        self.out_mode = AAFIOMode.Q_BNCHW_S_NBCHW

    def forward(self, features):
        """
        Arguments regrouped in features dict with at least the following keys 
        (it can change when align_first=False):
         - support_p1: support features aligned with query
                                List[Tensor] N_support x B x Channels x H x W
         - query_p1: query features aligned with support
                                List[Tensor] B x N_support x Channels x H x W
         - support_targets: targets boxes and label corresponding to each 
                            support imageList[BoxList]
        
        Returns:
         - support_p2: support features attention with query
                                List[Tensor] N_support x B x Channels x H x W
         - query_p2: query features attention with support
                                List[Tensor] B x N_support x Channels x H x W
        """
        pass


@registry.ATTENTION_MODULE.register("IDENTITY")
class AttentionIdentity(BaseAttention):
    def __init__(self, *args):
        super(AttentionIdentity, self).__init__(*args)
        self.in_mode = AAFIOMode.ID
        self.out_mode = AAFIOMode.ID


    def forward(self, features):
        """
        Arguments:
         - support_aligned_query: support features aligned with query
                                List[Tensor] N_support x B x Channels x H x W
         - query_aligned_support: query features aligned with support
                                List[Tensor] B x N_support x Channels x H x W
         - support_targets: targets boxes and label corresponding to each 
                            support imageList[BoxList]
        
        Returns:
         - support_attended_query: support features attention with query
                                List[Tensor] N_support x B x Channels x H x W
         - query_attended_support: query features attention with support
                                List[Tensor] B x N_support x Channels x H x W
        """
        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        features.update({
            'query' + self.output_name: query_features,
            'support' + self.output_name: support_features
        })



@registry.ATTENTION_MODULE.register("REWEIGHTING")
class AttentionRW(BaseAttention):
    def __init__(self, *args):
        super(AttentionRW, self).__init__(*args)

        self.pooled_vectors = None

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        N_s, B, C, _, _ = support_features[0].shape
        support_pooled = [
            # feat[:,0].max(-1)[0].max(-1)[0].reshape(N_s*C, 1, 1, 1)
            feat[:, 0].mean(dim=[-1, -2]).reshape(N_s*C, 1, 1, 1)
            for feat in support_features
        ]

        query_features = apply_tensor_list(query_features, 'flatten', 1, 2)
        support_attended_query = support_features
        query_attended_support = [

            F.conv2d(
                feat,
                support_pooled[level],
                groups=C*N_s
            ).reshape(B, N_s, C, feat.shape[-2], feat.shape[-1])
            for level, feat in enumerate(query_features)
        ]

        self.pooled_vectors = support_pooled
        self.support_target = support_targets


        features.update({
            'query' + self.output_name: query_attended_support,
            'support' + self.output_name: support_attended_query
        })


@registry.ATTENTION_MODULE.register("REWEIGHTING_BATCH")
class AttentionRWB(BaseAttention):
    def __init__(self, *args):
        super(AttentionRWB, self).__init__(*args)

        self.pooled_vectors = None

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        N_s, B, C, _, _ = support_features[0].shape

        K = self.cfg.FEWSHOT.K_SHOT

        N_way = N_s // K
        support_pooled = [
            # feat.permute(1, 0, 2, 3, 4).max(-1)[0].max(-1)[0].reshape(
            #     N_s * B * C, 1, 1, 1)
            # feat.permute(1,0,2,3,4).max(-1)[0].max(-1)[0].reshape(B, N_s, C, 1, 1)
            feat.permute(1, 0, 2, 3,
                         4).mean(dim=[-1, -2], keepdim=True)#.reshape(N_s * B * C, 1, 1, 1)
            for feat in support_features
        ]

        # query_features = apply_tensor_list(query_features, 'flatten', 0, 2)
        # # when using batched rw vectors
        # query_features = apply_tensor_list(query_features, 'unsqueeze', 0)
        support_attended_query = support_features
        query_attended_support = [
            # F.conv2d(feat, support_pooled[level],
            #          groups=C * N_s * B).reshape(B, N_way, K, C, feat.shape[-2],
            #                                        feat.shape[-1]).mean(dim=2)
            (feat * F.softmax(support_pooled[level], dim=2)
             ).reshape(B, N_way, K, C, feat.shape[-2],
                       feat.shape[-1]).mean(dim=2)
            for level, feat in enumerate(query_features)
        ]

        self.pooled_vectors = support_pooled
        self.support_target = support_targets
        self.query_attended_features = query_attended_support
        features.update({
            'query' + self.output_name: query_attended_support,
            'support' + self.output_name: support_attended_query
        })


@registry.ATTENTION_MODULE.register("REWEIGHTING_SMALL_SCALE")
class AttentionRWSS(BaseAttention):

    def __init__(self, *args):
        super(AttentionRWSS, self).__init__(*args)

        self.pooled_vectors = None

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        N_s, B, C, _, _ = support_features[0].shape

        K = self.cfg.FEWSHOT.K_SHOT

        N_way = N_s // K
        support_pooled = [
            # feat.permute(1, 0, 2, 3, 4).max(-1)[0].max(-1)[0].reshape(
            #     N_s * B * C, 1, 1, 1)
            # feat.permute(1,0,2,3,4).max(-1)[0].max(-1)[0].reshape(B, N_s, C, 1, 1)
            feat.permute(1, 0, 2, 3,
                         4).mean(dim=[-1, -2],
                                 keepdim=True)  #.reshape(N_s * B * C, 1, 1, 1)
            for feat in support_features
        ]

        support_pooled = [support_pooled[0]] * 3

        # query_features = apply_tensor_list(query_features, 'flatten', 0, 2)
        # # when using batched rw vectors
        # query_features = apply_tensor_list(query_features, 'unsqueeze', 0)
        support_attended_query = support_features
        query_attended_support = [
            # F.conv2d(feat, support_pooled[level],
            #          groups=C * N_s * B).reshape(B, N_way, K, C, feat.shape[-2],
            #                                        feat.shape[-1]).mean(dim=2)
            (feat * F.softmax(support_pooled[level], dim=2)
             ).reshape(B, N_way, K, C, feat.shape[-2],
                       feat.shape[-1]).mean(dim=2)
            for level, feat in enumerate(query_features[:3])
        ]

        self.pooled_vectors = support_pooled
        self.support_target = support_targets
        self.query_attended_features = query_attended_support[:3]
        features.update({
            'query' + self.output_name: query_attended_support,
            'support' + self.output_name: support_attended_query
        })


@registry.ATTENTION_MODULE.register("REWEIGHTING_WO_SOFT")
class AttentionRWWS(BaseAttention):
    def __init__(self, *args):
        super(AttentionRWWS, self).__init__(*args)

        self.pooled_vectors = None
        self.pool_size = 3



    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        N_s, B, C, _, _ = support_features[0].shape

        K = self.cfg.FEWSHOT.K_SHOT

        N_way = N_s // K
        # support_pooled = [
        #     # feat.permute(1, 0, 2, 3, 4).max(-1)[0].max(-1)[0].reshape(
        #     #     N_s * B * C, 1, 1, 1)
        #     # feat.permute(1,0,2,3,4).max(-1)[0].max(-1)[0].reshape(B, N_s, C, 1, 1)
        #     feat.permute(1, 0, 2, 3,
        #                  4).mean(dim=[-1, -2],
        #                          keepdim=True)  #.reshape(N_s * B * C, 1, 1, 1)
        #     for feat in support_features
        # ]

        support_pooled = []
        for level, feat in enumerate(support_features):
            N_S, B, C, H, W = feat.shape
            P = self.pool_size
            feat = feat.permute(1, 0, 2, 3, 4).reshape(B * N_s, C, H, W)
            targets = [t.bbox for t in support_targets] * B
            scale = 1/8 * 2 ** (- level)
            # pooled_feat = torchvision.ops.roi_align(feat, targets, output_size=P, spatial_scale=scale)
            # support_pooled.append(pooled_feat.mean(dim=[-1,-2], keepdim=True).reshape(B, N_s, C, 1, 1))
            support_pooled.append(
                feat.mean(dim=[-1, -2],
                                 keepdim=True).reshape(B, N_s, C, 1, 1))
            # support_pooled.append(
            #     pooled_feat.max(dim=-1, keepdim=True)[0].max(
            #                     dim=-2, keepdim=True)[0].reshape(B, N_s, C, 1, 1))
            # support_pooled.append(
            #     self.mlp(pooled_feat.reshape(-1, C * P * P)).reshape(
            #         B, N_s, C, 1, 1))

        def match_level(box_area):
            crop_area = self.cfg.FEWSHOT.SUPPORT.CROP_SIZE[0] ** 2
            # crop_area = crop_area.item()
            L = self.cfg.FEWSHOT.FEATURE_LEVEL - 1
            return min(L, max(0, math.floor(math.log(box_area / crop_area) + L)))

        target_labels = [match_level(t.area()) for t in support_targets]
        # print(target_labels)
        L = self.cfg.FEWSHOT.FEATURE_LEVEL
        level_masks = F.one_hot(torch.tensor(target_labels), num_classes=L).T.view(L, 1, N_s, 1, 1, 1).cuda()
        # for idx, level in enumerate(target_labels):
        #     for l in range(5):
        #         if level != l:
        #             support_pooled[l][:, idx, :] = 0
        # query_features = apply_tensor_list(query_features, 'flatten', 0, 2)
        # # when using batched rw vectors
        # query_features = apply_tensor_list(query_features, 'unsqueeze', 0)
        cos = torch.nn.CosineSimilarity(dim=2)
        support_attended_query = support_features
        query_attended_support = [
            # F.conv2d(feat, support_pooled[level],
            #          groups=C * N_s * B).reshape(B, N_way, K, C, feat.shape[-2],
            #                                        feat.shape[-1]).mean(dim=2)
            # (feat * F.softmax(support_pooled[level], dim=2)# * mask
            (
                # feat * torch.sigmoid(support_pooled[level])   # * mask
                feat * 0.5 * (1 + cos(feat, support_pooled[level]))
            ).reshape(B, N_way, K, C, feat.shape[-2],
                      feat.shape[-1]).mean(dim=2)
            for level, (feat,
                        mask) in enumerate(zip(query_features, level_masks))
        ]
        self.pooled_vectors = support_pooled
        self.support_target = support_targets
        self.query_attended_features = query_attended_support
        features.update({
            'query' + self.output_name: query_attended_support,
            'support' + self.output_name: support_attended_query
        })


@registry.ATTENTION_MODULE.register("SELF_ATTENTION")
class AttentionSelf(BaseAttention):
    def __init__(self, *args):
        super(AttentionSelf, self).__init__(*args)

        self.pooled_vectors = None

        self.level_attention = [nn.Conv2d(256, 1, 1) for i in range(5)]


    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        query_attended_support = []
        for feat, conv in zip(query_features, self.level_attention):
            attention_weight = F.sigmoid(conv(feat))
            query_attended_support.append(feat * attention_weight)


        features.update({
            'query' + self.output_name: query_attended_support,
            'support' + self.output_name: support_features
        })


@registry.ATTENTION_MODULE.register("BGA")
class BackgroundAttention(BaseAttention):
    def __init__(self, *args):
        super(BackgroundAttention, self).__init__(*args)

        self.pooled_vectors = None

        self.ba_block = BackgroundAttenuationBlock(256,5, self.cfg)

        self.in_mode = AAFIOMode.Q_BCHW_S_NCHW
        self.out_mode = AAFIOMode.Q_BCHW_S_NCHW


    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        support_features = self.ba_block(support_features)

        features.update({
            'query' + self.output_name: query_features,
            'support' + self.output_name: support_features
        })


@registry.ATTENTION_MODULE.register("META_FASTER")
class SimAttention(BaseAttention):
    def __init__(self, *args):
        super(SimAttention, self).__init__(*args)

        self.pooled_vectors = None


    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']
        attention_maps = features['attention_map']

        support_reweight = []
        query_reweight = []

        K = self.cfg.FEWSHOT.K_SHOT

        for level, feat in enumerate(support_features):
            B, Ns, C, H, W = query_features[level].shape
            attention_vector = F.softmax(attention_maps[level].sum(dim=-1), dim=-1)

            attention_vector = attention_vector.reshape(B, Ns, K, 1, H, W).mean(dim=2).to(
                self.cfg.MODEL.DEVICE)
            support_reweight.append(feat * attention_vector.permute(1, 0, 2, 3, 4))
            query_reweight.append(query_features[level] * attention_vector)
            del attention_vector

        features.update({
            'query' + self.output_name: query_reweight,
            'support' + self.output_name: support_reweight
        })

@registry.ATTENTION_MODULE.register("POOLING")
class PoolingAttention(BaseAttention):
    def __init__(self, *args):
        super(PoolingAttention, self).__init__(*args)

        self.pooling_size = 7

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        support_pooled = []
        query_pooled = []


        for level, feat in enumerate(support_features):
            B, Ns, C, H, W = query_features[level].shape

            box_tensor = [
                    torch.Tensor([0, 0, H, W]).unsqueeze(0).to(self.cfg.MODEL.DEVICE)
                        for _ in range(B*Ns)
            ]

            feat = feat.flatten(end_dim=1)
            query_feat = query_features[level].flatten(end_dim=1)


            pooled_support = torchvision.ops.roi_align(
                feat, box_tensor, output_size=self.pooling_size)
            pooled_support = pooled_support.reshape(Ns, B, C,
                                                    self.pooling_size,
                                                    self.pooling_size)

            pooled_query = torchvision.ops.roi_align(
                query_feat, box_tensor, output_size=self.pooling_size)
            pooled_query = pooled_query.reshape(B, Ns, C,
                                                    self.pooling_size,
                                                    self.pooling_size)
            support_pooled.append(pooled_support)
            query_pooled.append(pooled_query)


        features.update({
            'query' + self.output_name: query_pooled,
            'support' + self.output_name: support_pooled
        })


@registry.ATTENTION_MODULE.register("INTERPOLATE")
class InterpolateAttention(BaseAttention):
    def __init__(self, *args):
        super(InterpolateAttention, self).__init__(*args)


    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        support_pooled = []
        query_pooled = []

        for level, feat in enumerate(support_features):
            B, Ns, C, H, W = query_features[level].shape


            feat = feat.flatten(end_dim=1)

            support_inter = F.interpolate(feat, (H, W))


            support_pooled.append(support_inter.reshape(Ns, B, C, H, W))
            query_pooled.append(query_features[level])

        features.update({
            'query' + self.output_name: query_pooled,
            'support' + self.output_name: support_pooled
        })


@registry.ATTENTION_MODULE.register("GRU")
class GRUAttention(BaseAttention):
    def __init__(self, *args):
        super(GRUAttention, self).__init__(*args)
        self.relation_gru = RelationGRU(self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS)

        self.in_mode = AAFIOMode.Q_BCHW_S_NCHW
        self.out_mode = AAFIOMode.Q_BCHW_S_NCHW

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        support_pooled = []
        query_pooled = []
        K = self.cfg.FEWSHOT.K_SHOT

        for level, s_feat in enumerate(support_features):
            q_feat = query_features[level]
            B, C, H, W = q_feat.shape
            N = s_feat.shape[0]

            q_feat = q_feat.permute(0,2,3,1).flatten(end_dim=2) #BHW, C
            s_feat = s_feat.max(dim=-1)[0].max(dim=-1)[0] #NK, C
            s_feat = s_feat.unsqueeze(1).repeat(1, B*H*W, 1) # NK, BHW, C

            q_feat_attention = self.relation_gru(s_feat, q_feat) #NK, BHW, C
            q_feat_attention = q_feat_attention.reshape(N, B, H, W, C).permute(1,0,4,2,3) #B, NK, C, H, W

            q_feat_attention = q_feat_attention.reshape(B, N // K, K, C, H, W).mean(dim=2)

            support_pooled.append(s_feat)
            query_pooled.append(q_feat_attention)

        features.update({
            'query' + self.output_name: query_pooled,
            'support' + self.output_name: support_pooled
        })