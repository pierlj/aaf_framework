import torch
from torch import nn

from .... import registry
from .utils import AAFIOMode

class BaseFusionModule(nn.Module):
    def __init__(self, cfg, *args):
        super(BaseFusionModule, self).__init__(*args)

        self.cfg = cfg

        self.input_name = '_p2'
        self.output_name = 'output_features'

        self.in_mode = AAFIOMode.Q_BNCHW_S_NBCHW
        self.out_mode = AAFIOMode.Q_BNCHW

        self.channels = 256

    def apply_net(self, x, net, shape=None):
        """
        Function that wraps flatten unflatten ops around network forward. 
        """
        if shape is None:
            shape = x.shape
        x = x.flatten(end_dim=1)
        x = net(x)

        x = x.reshape(*shape)
        return x



@registry.FUSION_MODULE.register("IDENTITY")
class FusionIdentity(BaseFusionModule):
    def __init__(self, *args):
        super(FusionIdentity, self).__init__(*args)
        self.in_mode = AAFIOMode.ID
        self.out_mode = AAFIOMode.Q_BNCHW

    def forward(self, features):
        """
        Arguments:
         - support_attended_query: support features attention with query
                                List[Tensor] B x N_support x Channels x H x W
         - query_attended_support: query features attention with support
                                List[Tensor] N_support x B x Channels x H x W
         - support_targets: targets boxes and label corresponding to each 
                            support imageList[BoxList]
        
        Returns:
         - query_support_merged: support features aligned with query
                                List[Tensor] B x N_support x Channels x H x W

        """

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        features.update({self.output_name: query_features})


@registry.FUSION_MODULE.register("ADD")
class FusionAdd(BaseFusionModule):
    def __init__(self, *args):
        super(FusionAdd, self).__init__(*args)

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        query_support_merged = []
        for query, support in zip(query_features, support_features):
            Ns, B, C, Hs, Ws = support.shape
            B, Ns, C, Hq, Wq = query.shape

            assert Hs == Hq and Ws == Wq, 'Incompatible attention for this fusion module'

            feature_merged = query + support.permute(1, 0, 2, 3, 4)

            query_support_merged.append(feature_merged)

        features.update({self.output_name: query_support_merged})


@registry.FUSION_MODULE.register("HADAMARD")
class FusionHadamard(BaseFusionModule):
    def __init__(self, *args):
        super(FusionHadamard, self).__init__(*args)

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        query_support_merged = []
        for query, support in zip(query_features, support_features):
            Ns, B, C, Hs, Ws = support.shape
            B, Ns, C, Hq, Wq = query.shape

            assert Hs == Hq and Ws == Wq, 'Incompatible attention for this fusion module'

            feature_merged = query * support.permute(1, 0, 2, 3, 4)

            query_support_merged.append(feature_merged)

        features.update({self.output_name: query_support_merged})


@registry.FUSION_MODULE.register("SUBSTRACT")
class FusionSub(BaseFusionModule):
    def __init__(self, *args):
        super(FusionSub, self).__init__(*args)

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        query_support_merged = []
        for query, support in zip(query_features, support_features):
            Ns, B, C, Hs, Ws = support.shape
            B, Ns, C, Hq, Wq = query.shape

            assert Hs == Hq and Ws == Wq, 'Incompatible attention for this fusion module'

            feature_merged = query - support.permute(1, 0, 2, 3, 4)

            query_support_merged.append(feature_merged)

        features.update({self.output_name: query_support_merged})


@registry.FUSION_MODULE.register("CONCAT")
class FusionConcat(BaseFusionModule):
    def __init__(self, *args):
        super(FusionConcat, self).__init__(*args)

        # self.output_net = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
        #                                 nn.ReLU())

    def forward(self, features):

        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        query_support_merged = []
        for query, support in zip(query_features, support_features):
            Ns, B, C, Hs, Ws = support.shape
            B, Ns, C, Hq, Wq = query.shape

            assert Hs == Hq and Ws == Wq, 'Incompatible attention for this fusion module'

            # feature_merged = self.apply_net(torch.cat([query, support.permute(1, 0, 2, 3, 4)], dim=2),
            #                                 self.output_net,
            #                                 query.shape)
            feature_merged = torch.cat([query, support.permute(1, 0, 2, 3, 4)], dim=2)

            query_support_merged.append(feature_merged)


        features.update({self.output_name: query_support_merged})


@registry.FUSION_MODULE.register("META_FASTER")
class FusionMeta(BaseFusionModule):
    def __init__(self, *args):
        super(FusionMeta, self).__init__(*args)

        self.hadamard_net = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                            nn.ReLU())
        self.substract_net = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                           nn.ReLU())
        self.concat_net = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.ReLU())


    def forward(self, features):
        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        query_support_merged = []
        for query, support in zip(query_features, support_features):
            Ns, B, C, Hs, Ws = support.shape
            B, Ns, C, Hq, Wq = query.shape

            assert Hs == Hq and Ws == Wq, 'Incompatible attention for this fusion module'

            hadamard = self.apply_net(query * support.permute(1, 0, 2, 3, 4),
                        self.hadamard_net)
            substract = self.apply_net(query - support.permute(1, 0, 2, 3, 4),
                        self.substract_net)
            concat = self.apply_net(
                torch.cat([query, support.permute(1, 0, 2, 3, 4)], dim=2),
                self.concat_net)

            fusion_concat = torch.cat([hadamard, substract, concat], dim=2)
            # fusion_concat = fusion_concat.flatten(end_dim=1)
            # fusion_all = self.output_net(fusion_concat).reshape(B, Ns, C, Hq, Wq)
            # fusion_all = fusion_concat[:, :256, :, :].reshape(B, Ns, C, Hq, Wq)

            query_support_merged.append(fusion_concat)

        features.update({self.output_name: query_support_merged})


@registry.FUSION_MODULE.register("DYNAMIC_R")
class FusionDynamicR(BaseFusionModule):
    def __init__(self, *args):
        super(FusionDynamicR, self).__init__(*args)

        self.hadamard_net = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                          nn.ReLU())
        self.substract_net = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                           nn.ReLU())
        self.concat_net = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.ReLU())

    def forward(self, features):
        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        K = self.cfg.FEWSHOT.K_SHOT
        out_ch = self.hadamard_net[0].out_channels + \
                        self.substract_net[0].out_channels + \
                        self.concat_net[0].out_channels

        query_support_merged = []
        for query, support in zip(query_features, support_features):
            Ns, B, C, Hs, Ws = support.shape
            B, Ns, C, Hq, Wq = query.shape

            support = support.reshape(Ns // K, K, B, C, Hs, Ws).mean(dim=1)
            query = query.reshape(B, Ns // K, K, C, Hq, Wq).mean(dim=2)

            assert Hs == Hq and Ws == Wq, 'Incompatible attention for this fusion module'

            hadamard = self.apply_net(query * support.permute(1, 0, 2, 3, 4),
                                      self.hadamard_net)
            substract = self.apply_net(
                query - support.permute(1, 0, 2, 3, 4), self.substract_net)
            concat = self.apply_net(
                torch.cat([query, support.permute(1, 0, 2, 3, 4)], dim=2),
                self.concat_net)

            fusion_concat = torch.cat([hadamard, substract, concat], dim=2)

            query_support_merged.append(fusion_concat)

        features.update({self.output_name: query_support_merged})