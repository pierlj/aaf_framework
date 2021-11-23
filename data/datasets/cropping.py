import torch
from fcos_core.structures.bounding_box import BoxList
from ...utils.visualization import plot_single_img_boxes


class CroppingModule():
    """
    Cropping module manages how support examples are cropped and resized so 
    they can be processed by batch. 
    """
    def __init__(self, cfg, mode) :
        self.cfg = cfg
        self.mode = mode
        self.size = cfg.FEWSHOT.SUPPORT.CROP_SIZE
        self.margin = cfg.FEWSHOT.SUPPORT.CROP_MARGIN

    def crop(self, img, target):
        crop_method = getattr(self, 'crop_' + self.mode.lower())
        return crop_method(img, target)

    def crop_resize(self, img, target):
        """
        Resize target to a fixed size disregarding aspect ratio and original size
        """
        box = target.bbox[0]
        enlarged_box = box.clone()
        hw = box[2:] - box[:2]
        enlarged_box[:2] = enlarged_box[:2] - hw * self.margin
        enlarged_box[2:] = enlarged_box[2:] + hw * self.margin

        enlarged_box[::2] = enlarged_box[::2].clamp(0, img.shape[2])
        enlarged_box[1::2] = enlarged_box[1::2].clamp(0, img.shape[1])

        box_inside = box - enlarged_box[:2].repeat(2)
        enlarged_box = enlarged_box.long()
        img_cropped = img[:, enlarged_box[1]:enlarged_box[3],
                          enlarged_box[0]:enlarged_box[2]]

        area_dim = enlarged_box[2:] - enlarged_box[:2]
        target_inside = BoxList(box_inside.unsqueeze(0), area_dim)

        size = self.size
        if size is not None:

            # print(target_inside.size)
            target_inside = target_inside.resize(size)
            img_cropped = torch.nn.functional.interpolate(
                img_cropped.unsqueeze(0), size)[0]

        # Keep these in order to get the size of object w.r.t the original image
        target_inside.add_field('old_bbox', box)
        target_inside.add_field('old_img_size', img.shape)
        target_inside._copy_extra_fields(target)
        return img_cropped, target_inside

    def crop_keep_ratio(self, img, target):
        """
        Resize objects to support image size keeping ratio
        """
        box = target.bbox[0]

        enlarged_box = box.clone()
        hw = box[2:] - box[:2]
        new_box_size = max(self.size[0], hw.max())

        enlarged_box = self.get_enlarged_box(box, img.shape, margin=self.margin)

        box_inside = box - enlarged_box[:2].repeat(2)
        enlarged_box = enlarged_box.long()
        img_cropped = img[:, enlarged_box[1]:enlarged_box[3],
                          enlarged_box[0]:enlarged_box[2]]

        area_dim = enlarged_box[2:] - enlarged_box[:2]
        target_inside = BoxList(box_inside.unsqueeze(0), area_dim)

        size = self.size
        if size is not None:

            # print(target_inside.size)
            target_inside = target_inside.resize(size)
            img_cropped = torch.nn.functional.interpolate(
                img_cropped.unsqueeze(0), size)[0]

        # Keep these in order to get the size of object w.r.t the original image
        target_inside.add_field('old_bbox', box)
        target_inside.add_field('old_img_size', img.shape)
        target_inside._copy_extra_fields(target)
        return img_cropped, target_inside

    def crop_keep_size(self, img, target):
        """
        Keep size as in original img but cropped in maller size
        For object larger, resize is still required.
        """
        box = target.bbox[0].long()

        hw = box[2:] - box[:2]
        box[:2] = torch.clamp(box[:2] - self.margin * hw, 0)
        box[2:] = box[2:] + self.margin * hw
        box[2] = box[2].clamp(0, img.shape[-1])
        box[3] = box[3].clamp(0, img.shape[-2])

        enlarged_box = self.get_enlarged_box(box, img.shape, margin=0.0)

        box_inside = box - enlarged_box[:2].repeat(2)
        box_inside = box_inside.long()
        
        wh_enlarged = enlarged_box[2:] - enlarged_box[:2]
        box_inside[::2] += (wh_enlarged[0] / 2 - torch.mean(box_inside[::2].float())).long()
        box_inside[1::2] += (wh_enlarged[1] / 2 - torch.mean(box_inside[1::2].float())).long()


        enlarged_box = enlarged_box.long()
        enlarged_hw = enlarged_box[2:] - enlarged_box[:2]
        img_cropped = torch.zeros(img.shape[0], *enlarged_hw.flip(0))
        img_cropped[:, box_inside[1]:box_inside[3],box_inside[0]:box_inside[2]] = img[:, box[1]:box[3],box[0]:box[2]]

        area_dim = enlarged_box[2:] - enlarged_box[:2]
        target_inside = BoxList(box_inside.unsqueeze(0), area_dim)
        size = self.size
        if size is not None:

            # print(target_inside.size)
            target_inside = target_inside.resize(size)
            img_cropped = torch.nn.functional.interpolate(
                img_cropped.unsqueeze(0), size)[0]

        # Keep these in order to get the size of object w.r.t the original image
        target_inside.add_field('old_bbox', box)
        target_inside.add_field('old_img_size', img.shape)
        target_inside._copy_extra_fields(target)
        return img_cropped, target_inside

    def get_enlarged_box(self, box, shape, margin=0.0):
        enlarged_box = box.clone()
        hw = box[2:] - box[:2]
        new_box_size = max(self.size[0], hw.max())

        shape = torch.tensor(shape)

        enlarged_box[:2] = torch.max(torch.zeros_like(box[:2]), box[:2] - (new_box_size - hw)/2) + \
                            torch.min(torch.zeros_like(box[:2]), shape[-2:].flip(0) - (box[2:] + (new_box_size - hw) / 2 )) - \
                            hw * margin
        enlarged_box[2:] = torch.min(torch.ones_like(box[:2]) * shape[-2:].flip(0), box[2:] + (new_box_size - hw)/2) + \
                            torch.max(torch.zeros_like(box[:2]),(new_box_size - hw) /2 - box[:2]) + \
                            hw * margin

        enlarged_box[::2] = enlarged_box[::2].clamp(0, shape[2])
        enlarged_box[1::2] = enlarged_box[1::2].clamp(0, shape[1])

        return enlarged_box