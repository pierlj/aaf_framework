# AAF framework 

## Framework generalities
This repository contains the code of the AAF framework proposed in this
[paper](https://pierlj.github.io/). The main idea behind this work is to propose
a flexible framework to implement various attention mechanisms for Few-Shot
Object Detection. The framework is composed of 3 different modules: Spatial Alignment, 
Global Attention and Fusion Layer, which are applied successively to combine 
features from query and support images. 

The inputs of the framework are:
- query_features `List[Tensor(B, C, H, W)]`: Query features at different levels. For each level, the features are of shape Batch x Channels x Height x Width.
- support_features `List[Tensor(N, C, H', W')]` : Support features at different level. First dimension correspond to the number of support images, regrouped by class: `N = N_WAY * K_SHOT`.
- support_targets `List[BoxList]` bounding boxes for object in each support image. 

The framework can be configured using a separate config file. Examples of such files are available under `/config_files/aaf_framework/`. The structure of these files is simple: 
```python
ALIGN_FIRST: #True/False Run Alignment before Attention when True
ALIGNMENT:
    MODE: # Name of the alignment module selected
ATTENTION:
    MODE: # Name of the attention module selected
FUSION:
    MODE: # Name of the fusion module selected
```
| File name                | Method                           | Alignment          | Attention         | Fusion      |
|--------------------------|----------------------------------|--------------------|-------------------|-------------|
| `identity.yaml`            | Identity                         | IDENTITY           | IDENTITY          | IDENTITY    |
| `feature_reweighting.yaml` | [FSOD via feature reweighting](https://arxiv.org/pdf/1812.01866v2.pdf)     | IDENTITY           | REWEIGHTING_BATCH | IDENTITY    |
| `meta_faster_rcnn.yaml`    | [Meta Faster-RCNN](https://arxiv.org/pdf/2104.07719.pdf)                 | SIMILARITY_ALIGN   | META_FASTER       | META_FASTER |
| `self_adapt.yaml`          | [Self-adaptive attention for FSOD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9426416) | IDENTITY_NO_REPEAT | GRU               | IDENTITY    |
| `dynamic.yaml`             | [Dynamic relevance learning](https://arxiv.org/pdf/2108.02235.pdf)       | IDENTITY           | INTERPOLATE       | DYNAMIC_R   |
| `dana.yaml`                | [Dual Awarness Attention for FSOD](https://arxiv.org/pdf/2102.12152v3.pdf) | CISA               | BGA               | HADAMARD    |

The path to the AAF config file should be specified inside the master config file (i.e. for the whole network) under `FEWSHOT.AAF.CFG`. 

For each module, classes implementing the available choices are regrouped under a single file: `/modelling/aaf/alignment.py`, `/modelling/aaf/attention.py` and `/modelling/aaf/fusion.py`.
### Spatial Alignment
Spatial Alignment reorganizes spatially the features of one feature map to match another one. The idea is to align similar features in both maps so that comparison is easier.

| Name               | Description                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------|
| IDENTITY           | Repeats the feature to match BNCHW and NBCHW dimensions                                      |
| IDENTITY_NO_REPEAT | Identity without repetition                                                                  |
| SIMILARITY_ALIGN   | Compute similarity matrix between support  and query and align support to query accordingly. |
| CISA               | CISA block from [this method](https://arxiv.org/pdf/2102.12152.pdf)                          |

### Global Attention
Global Attention highlights some features of a map accordingly to an attention vector computed globally on another one. The idea is to leverage global and hopefully semantic information. 

| Name              | Description                                                                                            |
|-------------------|--------------------------------------------------------------------------------------------------------|
| IDENTITY          | Simply pass features to next modules.                                                                  |
| REWEIGHTING       | Reweights query features using globally pooled vectors from support.                                   |
| REWEIGHTING_BATCH | Same as above but support examples are the same  for the whole batch.                                  |
| SELF_ATTENTION    | Same as above but attention vectors are computed  from the alignment matrix between query and support. |
| BGA               | BGA blocks from [this method](https://arxiv.org/pdf/2102.12152.pdf)                                   |
| META_FASTER       | Attention block from [this method](https://arxiv.org/abs/2104.07719)                              |
| POOLING           | Pools query and support features to the same size.                                                     |
| INTERPOLATE       | Upsamples support features to match query size.                                                        |
| GRU               | Computes attention vectors through a graph  representation using a GRU.                                |
### Fusion Layer
Combine directly the features from support and query. These maps must be of the same dimension for point-wise operation. Hence fusion is often employed along with alignment. 

| Name        | Description                                                        |
|-------------|--------------------------------------------------------------------|
| IDENTITY    | Returns onlu adapted query features.                               |
| ADD         | Point-wise sum between query and support features.                 |
| HADAMARD    | Point-wise multiplication between query and support features.      |
| SUBSTRACT   | Point-wise substraction between query and support features.        |
| CONCAT      | Channel concatenation of query and support features.               |
| META_FASTER | Fusion layer from [this method](https://arxiv.org/abs/2104.07719)  |
| DYNAMIC_R   | Fusion layer from  [this method](https://arxiv.org/abs/2108.02235) |

## Training and evaluation
Training and evaluation scripts are available. 

TODO: Give code snippet to run training with a specified config file (modify main)
Basically create 2 scripts train.py and eval.py with arg config file.

## DataHandler 
Explain `DataHandler` class a bit. 
## Installation 

Dependencies used for this projects can be installed through `conda create --name <env> --file requirements.txt`. 
Please note that these requirements are not all necessary and it will be updated soon. 

FCOS must be installed from sources. But there might be some issue after installation depending
on the version of the python packages you use. 

- `cpu/vision.h` file not found: replace all occurences in the FCOS source by `vision.h` (see this [issue](https://github.com/tianzhi0549/FCOS/issues/351)). 
- Error related to `AT_CHECK` with pytorch > 1.5 : replace all occurences by `TORCH_CHECK` (see this [issue](https://github.com/tianzhi0549/FCOS/issues/357).
- Error related to `torch._six.PY36`: replace all occurence of `PY36` by `PY37`.

## Results
Results on pascal VOC, COCO and DOTA and DIOR. 



