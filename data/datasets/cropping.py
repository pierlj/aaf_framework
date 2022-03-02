import torch
from fcos_core.structures.bounding_box import BoxList
from ...utils.visualization import plot_single_img_boxes
from ...utils.utils import pad_to_size
from ..transforms import Cutout
import random

class CroppingModule():
    """
    Cropping module manages how support examples are cropped and resized so 
    they can be processed by batch. 
    """
    def __init__(self, cfg, mode, is_train) :
        self.cfg = cfg
        self.mode = mode
        self.is_train = is_train
        self.size = cfg.FEWSHOT.SUPPORT.CROP_SIZE
        self.margin = cfg.FEWSHOT.SUPPORT.CROP_MARGIN
        self.center_crop = cfg.FEWSHOT.SUPPORT.CROP_CENTER

        self.min_size = 32
        self.thickness = 0.5

        self.cutout = Cutout(cfg.AUGMENT.CUTOUT_PROBA_SUPPORT, cfg.AUGMENT.CUTOUT_SCALE)

        self.ms_cropper = MultiScaleSupport(margin=self.margin)

    def crop(self, img, target):
        crop_method = getattr(self, 'crop_' + self.mode.lower())
        return crop_method(img, target)

    def crop_resize(self, img, target):
        """
        Resize target to a fixed size preserving aspect ratio
        """
        box = target.bbox[0]
        enlarged_box = box.clone()
        hw = box[2:] - box[:2]
        max_dim = torch.ones_like(hw) * hw.max()

        enlarged_box[:2] = enlarged_box[:2] - (max_dim - hw) / 2
        enlarged_box[2:] = enlarged_box[2:] + (max_dim - hw) / 2

        enlarged_box[:2] = enlarged_box[:2] - max_dim * self.margin
        enlarged_box[2:] = enlarged_box[2:] + max_dim * self.margin

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

    def crop_keep_size_no_pad(self, img, target):
        """
        Resize objects to support image size keeping ratio.
        Objects smaller than self.size are kept to the same size.

        """
        box = target.bbox[0].long()

        hw = box[2:] - box[:2]
        box[:2] = box[:2] - self.margin * hw
        box[2:] = box[2:] + self.margin * hw

        box[::2] = box[::2].clamp(0, img.shape[2])
        box[1::2] = box[1::2].clamp(0, img.shape[1])

        enlarged_box = self.get_enlarged_box(box, img.shape, margin=0.0)

        box_inside = box - enlarged_box[:2].repeat(2)
        box_inside = box_inside.long()

        # Center crop
        # if self.center_crop:
        #     wh_enlarged = enlarged_box[2:] - enlarged_box[:2]
        #     box_inside[::2] += (wh_enlarged[0] / 2 -
        #                         torch.mean(box_inside[::2].float())).long()
        #     box_inside[1::2] += (wh_enlarged[1] / 2 -
        #                          torch.mean(box_inside[1::2].float())).long()

        enlarged_box = enlarged_box.long()
        enlarged_hw = enlarged_box[2:] - enlarged_box[:2]
        # img_cropped = torch.zeros(img.shape[0], *enlarged_hw.flip(0))
        img_cropped = img[:, enlarged_box[1]:enlarged_box[3], enlarged_box[0]:enlarged_box[2]]

        area_dim = enlarged_box[2:] - enlarged_box[:2]
        target_inside = BoxList(box_inside.unsqueeze(0), area_dim)

        if self.size is not None:
            target_inside = target_inside.resize(self.size)
            img_cropped = torch.nn.functional.interpolate(
                img_cropped.unsqueeze(0), self.size)[0]

        # Keep these in order to get the size of object w.r.t the original image
        target_inside.add_field('old_bbox', box)
        target_inside.add_field('old_img_size', img.shape)
        target_inside._copy_extra_fields(target)
        return img_cropped, target_inside

    def crop_keep_size(self, img, target):
        """
        Resize objects to support image size keeping ratio.
        Objects smaller than self.size are kept to the same size.

        Discard context outside enlarged_box (zero_pad)
        """
        box = target.bbox[0].long()

        hw = box[2:] - box[:2]
        box[:2] = box[:2] - self.margin * hw
        box[2:] = box[2:] + self.margin * hw

        box[::2] = box[::2].clamp(0, img.shape[2])
        box[1::2] = box[1::2].clamp(0, img.shape[1])

        enlarged_box = self.get_enlarged_box(box, img.shape, margin=0.0)

        box_inside = box - enlarged_box[:2].repeat(2)
        box_inside = box_inside.long()

        # Center crop
        if self.center_crop:
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


    def crop_reflect(self, img, target):
        """
        Resize objects to support image size keeping ratio.
        Objects smaller than self.size are kept to the same size.

        Discard context outside enlarged_box (miror_pad)
        """
        box = target.bbox[0].long()

        hw = box[2:] - box[:2]
        max_dim = torch.ones_like(hw) * hw.max()

        enlarged_box = box.clone()
        enlarged_box[:2] = enlarged_box[:2] - (max_dim - hw) / 2
        enlarged_box[2:] = enlarged_box[2:] + (max_dim - hw) / 2

        enlarged_box[:2] = enlarged_box[:2] - max_dim * self.margin
        enlarged_box[2:] = enlarged_box[2:] + max_dim * self.margin

        enlarged_box[::2] = enlarged_box[::2].clamp(0, img.shape[2])
        enlarged_box[1::2] = enlarged_box[1::2].clamp(0, img.shape[1])
        enlared_hw = enlarged_box[2:] - enlarged_box[:2]

        box_inside = box - enlarged_box[:2].repeat(2)
        reflect_box = box_inside.clone()
        reflect_box[:2] = reflect_box[:2] - hw * self.margin / 2
        reflect_box[2:] = reflect_box[2:] + hw * self.margin / 2
        reflect_box[::2] = reflect_box[::2].clamp(0, enlared_hw[0])
        reflect_box[1::2] = reflect_box[1::2].clamp(0, enlared_hw[1])

        area_dim = enlarged_box[2:] - enlarged_box[:2]
        target_inside = BoxList(box_inside.unsqueeze(0), area_dim)
        reflect_box = BoxList(reflect_box.unsqueeze(0), area_dim)

        img_cropped = img[:, enlarged_box[1]:enlarged_box[3],
                          enlarged_box[0]:enlarged_box[2]]


        # if enlarged_box is wider than support image size resize it
        if self.size[0] < enlared_hw[0] or self.size[1] < enlared_hw[1]:
            target_inside = target_inside.resize(self.size)
            reflect_box = reflect_box.resize(self.size)
            img_cropped = torch.nn.functional.interpolate(
                img_cropped.unsqueeze(0), self.size)[0]

        reflect_box = reflect_box.bbox[0].long()
        img_cropped = img_cropped[:, reflect_box[1]:reflect_box[3],
                                  reflect_box[0]:reflect_box[2]]

        target_inside_reflect = target_inside.bbox[0] - reflect_box[:2].repeat(2)
        img_cropped, box_inside = pad_to_size(img_cropped, self.size,
                                                 target_inside_reflect)

        img_cropped = self.cutout(img_cropped)
        target_inside.bbox = box_inside.unsqueeze(0)
        # Keep these in order to get the size of object w.r.t the original image
        target_inside.add_field('old_bbox', box)
        target_inside.add_field('old_img_size', img.shape)
        target_inside._copy_extra_fields(target)
        return img_cropped, target_inside



    def crop_resize_mask(self, img, target):
        """
        Resize target to a fixed size but mask out the object.

        EXPERIMENTATION PURPOSE
        Testing if network can learn shortcuts from background/context
        """
        box = target.bbox[0].long()
        img[:,box[1]:box[3], box[0]:box[2]] = 0
        enlarged_box = box.clone()
        hw = box[2:] - box[:2]
        max_dim = torch.ones_like(hw) * hw.max()

        enlarged_box[:2] = enlarged_box[:2] - (max_dim - hw) / 2
        enlarged_box[2:] = enlarged_box[2:] + (max_dim - hw) / 2

        enlarged_box[:2] = enlarged_box[:2] - max_dim * self.margin
        enlarged_box[2:] = enlarged_box[2:] + max_dim * self.margin

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

    def crop_multi_scale(self, image, target):
        images, targets = self.ms_cropper(image, target)
        images = [
            torch.nn.functional.interpolate(img.unsqueeze(0), self.size)[0]
                for img in images
        ]

        targets = [t.resize(self.size) for t in targets]
        target.add_field('ms_targets', targets)
        image = torch.cat(images, dim=0)
        return image, target

    def crop_adaptive(self, img, target):
        """
        Resize target to a random size preserving aspect ratio
        Random size is sampled from a range depending of object size.  
        """

        box = target.bbox[0]
        enlarged_box = box.clone()
        hw = box[2:] - box[:2]
        max_dim = torch.ones_like(hw) * hw.max()

        enlarged_box[:2] = enlarged_box[:2] - (max_dim - hw) / 2
        enlarged_box[2:] = enlarged_box[2:] + (max_dim - hw) / 2

        if not (self.is_train and self.cfg.FEWSHOT.SUPPORT.CROP_AUG):
            sampled_size = min(self.size[0], max(self.min_size, max_dim[0]))
        else:
            s_plus = min(self.size[0],
                        max(self.min_size, (1 + self.thickness) *max_dim[0]))
            s_minus = min(self.size[0],
                        max(self.min_size, (1 - self.thickness) * max_dim[0]))

            u = random.random()
            sampled_size = s_minus + u * (s_plus - s_minus)

        crop_size = self.size[0] * max_dim / sampled_size


        enlarged_box[:2] = enlarged_box[:2] - (crop_size - max_dim) / 2
        enlarged_box[2:] = enlarged_box[2:] + (crop_size - max_dim) / 2


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

    def crop_mixed(self, image, target):
        if target.area()[0].sqrt() <= 32:
            return self.crop_keep_size(image, target)
        else:
            return self.crop_resize(image, target)

    def get_enlarged_box(self, box, shape, margin=0.0):
        enlarged_box = box.clone()
        hw = box[2:] - box[:2]
        new_box_size = max(self.size[0], hw.max())

        shape = torch.tensor(shape)

        pixel_shift = (new_box_size - hw) / 2
        random_d = torch.rand(2) * 2 - 1 # get random shift between -1 and 1
        delta = pixel_shift * random_d # random pixel shift for the box inside enlarged one

        enlarged_box[:2] = torch.max(torch.zeros_like(box[:2]), box[:2] - (pixel_shift + delta)) + \
                            torch.min(torch.zeros_like(box[:2]), shape[-2:].flip(0) - (box[2:] + pixel_shift - delta)) - \
                            hw * margin
        enlarged_box[2:] = torch.min(torch.ones_like(box[:2]) * shape[-2:].flip(0), box[2:] + pixel_shift - delta) + \
                            torch.max(torch.zeros_like(box[:2]),pixel_shift + delta - box[:2]) + \
                            hw * margin

        enlarged_box[:2] = enlarged_box[:2]

        enlarged_box[::2] = enlarged_box[::2].clamp(0, shape[2])
        enlarged_box[1::2] = enlarged_box[1::2].clamp(0, shape[1])

        return enlarged_box


class MultiScaleSupport(object):
    def __init__(self, min_size=32, step=2, n_boxes=3, margin=0.0):
        self.min_size = min_size
        self.step = step
        self.n_boxes = n_boxes
        self.margin = margin

        self.object_sizes = [1, 4, 8] # [256,96,32]

        self.extract_context = True

    def __call__(self, image, target=None):
        if target is None:
            return image
        else:
            box = target.bbox[0]
            hw = box[2:] - box[:2]
            largest_dim = hw.max()

            box[:2] = box[:2] - largest_dim * self.margin
            box[2:] = box[2:] + largest_dim * self.margin

            hw = box[2:] - box[:2]
            center = ((box[2:] + box[:2]) / 2).long()
            largest_dim = hw.max()

            ms_boxes = []
            for i in range(self.n_boxes):
                delta = int(self.object_sizes[i] * max(self.min_size, largest_dim) // 2)
                ms_boxes.append(torch.cat([center - delta, center + delta]))
            ms_boxes.reverse() # get small object first, i.e. large box first
            images_cropped, targets = self.crop(image, ms_boxes, target)

            return images_cropped, targets  # concat in channel dimension

    def crop(self, img, boxes, target):
        images = []
        targets = []
        C, H, W = img.shape
        for box in boxes:
            box_clamped = box.clone()
            box_clamped[::2] = box_clamped[::2].clamp(0, W)
            box_clamped[1::2] = box_clamped[1::2].clamp(0, H)
            box = box.long()
            box_clamped = box_clamped.long()
            box_hw = box[2:] - box[:2]

            pad = (box * torch.tensor([-1, -1, 1, 1]) - torch.tensor([0, 0, *img.shape[-2:]]))
            h_pad, w_pad = pad[1::2].long(), pad[::2].long()
            frame = torch.zeros(C, *box_hw)

            h_slice_frame = slice(box_clamped[1] + h_pad[0], box_clamped[3] + h_pad[0])
            w_slice_frame = slice(box_clamped[0] + w_pad[0], box_clamped[2] + w_pad[0])
            frame[:, h_slice_frame, w_slice_frame] = \
                      img[:, box_clamped[1]:box_clamped[3],
                             box_clamped[0]:box_clamped[2]]

            current_target = target.copy_with_fields(target.fields())
            current_target.bbox = (current_target.bbox - box[:2].repeat(2))
            current_target.size = box[2:] - box[:2]
            if self.extract_context:
                frame_no_ctx = torch.zeros(C, *box_hw)
                obj_box = current_target.bbox[0].long()
                frame_no_ctx[:, obj_box[1]:obj_box[3], obj_box[0]:
                            obj_box[2]] = frame[:, obj_box[1]:obj_box[3],
                                                obj_box[0]: obj_box[2]]
                frame = frame_no_ctx
            images.append(frame)
            targets.append(current_target)
        return images, targets