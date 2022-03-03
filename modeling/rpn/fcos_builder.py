from .fcos.fcos import FCOSModule
from .fsfcos.fcos import FSFCOSModule
from .proto_fcos.fcos import ProtoFCOSModule

def build_fcos(cfg, in_channels):
    if cfg.FEWSHOT.ENABLED:
        if cfg.FEWSHOT.USE_PROTO_CLASSIFIER:
            return ProtoFCOSModule(cfg, in_channels)
        else:
            return FSFCOSModule(cfg, in_channels)
    else:
        return FCOSModule(cfg, in_channels)