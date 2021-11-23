import os

from yacs.config import CfgNode as CN

_C = CN()
# Run Alignment before Attention when True
_C.ALIGN_FIRST = True
_C.OUT_CH = 256

_C.ALIGNMENT = CN()
# Available options depend on the content modeling/rpn/fsfcos/aaf/alignment.py
_C.ALIGNMENT.MODE = 'IDENTITY'


_C.ATTENTION = CN()
# Available options depend on the content modeling/rpn/fsfcos/aaf/attention.py
_C.ATTENTION.MODE = 'IDENTITY'


_C.FUSION = CN()
# Available options depend on the content modeling/rpn/fsfcos/aaf/fusion.py
_C.FUSION.MODE = 'IDENTITY'
