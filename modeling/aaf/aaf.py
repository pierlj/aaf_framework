import torch
from torch import nn

from .. import registry
from ...config.aaf_module import cfg as aaf_cfg
from .utils import AAFIOMode, prime_factors

class AAFModule(nn.Module):
    def __init__(self, cfg, *args):
        """
        Alignment Attention Fusion Module apply successively each of its components:
            - Alignment
            - Attention 
            - Fusion
        
        It combines query features with support features for few-shot applications. The resulting
        class-specific query features can then be used for any few-shot task (classification, 
        detection, segmentation, etc.). 

        """
        super(AAFModule, self).__init__(*args)

        self.cfg = cfg

        aaf_cfg.merge_from_file(cfg.FEWSHOT.AAF.CFG)
        self.aaf_cfg = aaf_cfg

        self.alignment = registry.ALIGNMENT_MODULE[aaf_cfg.ALIGNMENT.MODE](cfg, aaf_cfg.ALIGN_FIRST)
        self.attention = registry.ATTENTION_MODULE[aaf_cfg.ATTENTION.MODE](cfg, aaf_cfg.ALIGN_FIRST)
        self.fusion = registry.FUSION_MODULE[aaf_cfg.FUSION.MODE](cfg)

        assert self.check_modules_compatibility(), 'Wrong combination of AAF modules, please change cfg file.'

    def forward(self, query_features, support_features, support_targets):
        """
        Arguments:
            - query_features: features extracted from the query images (a batch) 
                        List[Tensor(B x C x H x W)] 
            - support_features: features extracted from the support images 
                        Note that N is the number of support images and is a multiple of 
                        N = N_ways * K_shot (when SAME_SUPPORT_IN_BATCH=False N = B * N_ways * K_shot)
                        List[Tensor(N x C x H' x W')] 
            - support_targets: targets coming along with support features List[BoxList]
        
        Returns:
            - query_features_dict: dict containing class-specific query features
                        Dict[class: List[Tensor(B x N x C x H x W)]]
        
        For analysis purpose a dict features containing temporary results (in between modules) is
        used for throughout the module. That way, output of each submodule can be observed and analyzed. 

        Note that it actually outputs instance-specific query feature regrouped by class. 
        """
        features = {
            'query_features': query_features,
            'support_features': support_features,
            'support_targets': support_targets
        }
        if self.aaf_cfg.ALIGN_FIRST:
            self.alignment(features)
            self.attention(features)
        else:
            self.attention(features)
            self.alignment(features)

        self.fusion(features)

        labels = sorted(list(set([
            target.get_field('labels')[0].item()
            for target in support_targets
        ])))

        k = self.cfg.FEWSHOT.K_SHOT
        n_ways = len(labels)
        query_features_dict = {}
        # print(labels)
        for feature in features['output_features']:
            shape = feature.shape
            feature = feature.reshape(shape[0], n_ways, *shape[-3:])
            for id_class, c in enumerate(labels):
                if c not in query_features_dict:
                    query_features_dict[c] = []
                query_features_dict[c].append(feature[:,id_class])

        return query_features_dict

    def check_modules_compatibility(self):
        """
        Check if selected aaf modules are comptabile two by two. 

        Each module as two attributes:
            - in_mode: data structure expected as input
            - out_mode: data structure expected as output
        
        Modes can take values from AAFIOMode's attributes. 
        """
        if self.aaf_cfg.ALIGN_FIRST:
            if self.alignment.in_mode not in [AAFIOMode.Q_BCHW_S_NCHW, AAFIOMode.ID]:
                print('Incompatible Alignment module')
                return False
            p1 = self.attention.in_mode * self.alignment.out_mode
            p2 = self.fusion.in_mode * self.attention.out_mode
        else:
            if self.attention.in_mode not in [AAFIOMode.Q_BCHW_S_NCHW, AAFIOMode.ID]:
                print('Incompatible Attention module')
                return False
            p1 = self.alignment.in_mode * self.attention.out_mode
            p2 = self.fusion.in_mode * self.alignment.out_mode

        # overcomplicated way to check compatibility between modules
        # but can be extended with more than 2 modes
        if len(set(prime_factors(p1))) > 1 or len(set(prime_factors(p2))) > 1:
            return False

        if self.fusion.out_mode != AAFIOMode.Q_BNCHW:
            print('Incompatible Fusion module') # that case should never happen
            return False

        return True
