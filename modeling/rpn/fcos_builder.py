from .fcos.fcos import FCOSModule
from .fsfcos.fcos import FSFCOSModule

def build_fcos(cfg, in_channels):
    if cfg.FEWSHOT.ENABLED:
        return FSFCOSModule(cfg, in_channels)
    else:
        return FCOSModule(cfg, in_channels)