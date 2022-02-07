# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    """
    Add here the paths to customs dataset. They should be either COCO or VOC format.

    For COCO: 'img_dir'=path/to/image/folder and 'ann_file'=path/to/annotation/file
    Add one entry for each split (train, val, test).

    For VOC: check original repository for more documentation: 
    https://github.com/tianzhi0549/FCOS/tree/master.
    """
    DATA_DIR = "/home/pierre/Documents/PHD/Datasets"
    DATASETS = {
        "dota_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DOTA_v2/coco_format/train2017",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DOTA_v2/coco_format/annotations/instances_train2017.json"
        },
        "dota_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DOTA_v2/coco_format/val2017",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DOTA_v2/coco_format/annotations/instances_val2017.json"
        },
        "dota_test": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DOTA_v2/coco_format/test2017",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DOTA_v2/coco_format/annotations/instances_test2017.json"
        },
        "coco_2017_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/MSCOCO/train2017",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/MSCOCO/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/MSCOCO/val2017",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/MSCOCO/annotations/instances_val2017.json"
        },
        "coco_2017_test_dev": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/MSCOCO/test2017",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/MSCOCO/annotations/image_info_test-dev2017.json"
        },
        "pascalv_2012_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2012/train",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2012/annotations/instances_train.json"
        },
        "pascalv_2012_test": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2012/test",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2012/annotations/instances_test.json"
        },
        "pascalv_2012_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2012/val",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2012/annotations/instances_val.json"
        },
        "pascalv_2007_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2007/train",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2007/annotations/instances_train.json"
        },
        "pascalv_2007_test": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2007/test",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2007/annotations/instances_test.json"
        },
        "pascalv_2007_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2007/val",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/2007/annotations/instances_val.json"
        },
        "pascalv_merged_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/Merged/train",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/Merged/annotations/instances_train.json"
        },
        "pascalv_merged_test": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/Merged/test",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/Merged/annotations/instances_test.json"
        },
        "pascalv_merged_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/Merged/val",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/Merged/annotations/instances_val.json"
        },
        "voc_2012_train": {
            "data_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_2012/VOCdevkit/VOC2012",
            "split": "train"
        },
        "voc_2012_val": {
            "data_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_2012/VOCdevkit/VOC2012",
            "split": "val"
        },
        "voc_2012_test": {
            "data_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_2012/VOCdevkit/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "voc_2007_train": {
            "data_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_2007/VOCdevkit/VOC2007",
            "split": "train"
        },
        "voc_2007_val": {
            "data_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_2007/VOCdevkit/VOC2007",
            "split": "val"
        },
        "voc_2007_test": {
            "data_dir":
            "/home/pierre/Documents/PHD/Datasets/VOC_2007/VOCdevkit/VOC2007",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "vhr_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VHR_10/coco_format/train",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VHR_10/coco_format/annotations/instances_train.json"
        },
        "vhr_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VHR_10/coco_format/test",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VHR_10/coco_format/annotations/instances_test.json"
        },
        "vhr_test": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/VHR_10/coco_format/test",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/VHR_10/coco_format/annotations/instances_test.json"
        },
        "dior_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DIOR/coco_format/train",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DIOR/coco_format/annotations/instances_train.json"
        },
        "dior_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DIOR/coco_format/val",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DIOR/coco_format/annotations/instances_val.json"
        },
        "dior_test": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DIOR/coco_format/test",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DIOR/coco_format/annotations/instances_test.json"
        },
        "dota_256_train": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DOTA_256/coco_format/train",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DOTA_256/coco_format/annotations/instances_train.json"
        },
        "dota_256_val": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DOTA_256/coco_format/val",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DOTA_256/coco_format/annotations/instances_val.json"
        },
        "dota_256_test": {
            "img_dir":
            "/home/pierre/Documents/PHD/Datasets/DOTA_256/coco_format/test",
            "ann_file":
            "/home/pierre/Documents/PHD/Datasets/DOTA_256/coco_format/annotations/instances_test.json"
        }
    }

    @staticmethod
    def get(name):
        # Changed from original implementation to use only coco format
        # if 'coco' in name or 'dota' in name or 'pascalv' in name:
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["img_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
        )
        return dict(
            factory="COCODataset",
            args=args,
        )
        # elif "voc" in name:
        #     data_dir = DatasetCatalog.DATA_DIR
        #     attrs = DatasetCatalog.DATASETS[name]
        #     args = dict(
        #         data_dir=os.path.join(data_dir, attrs["data_dir"]),
        #         split=attrs["split"],
        #     )
        #     return dict(
        #         factory="PascalVOCDataset",
        #         args=args,
        #     )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d":
        "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/20171220/X-101-64x4d":
        "ImageNetPretrained/20171220/X-101-64x4d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x":
        "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(
            dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join(
            [prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
