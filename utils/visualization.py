import numpy as np
import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors


def plot_single_img_boxes(image, boxes_list, cfg, backend=None, threshold=0.0, bgr=False):
    if backend is not None:
        matplotlib.use(backend)

    scale = 1
    if cfg.INPUT.PIXEL_MEAN[0] > 1:
        scale = 255
    img = (image.permute(1,2,0).cpu().numpy() * cfg.INPUT.PIXEL_STD + cfg.INPUT.PIXEL_MEAN) / scale

    box_tensor = boxes_list.bbox
    labels = boxes_list.get_field('labels')
    colors = ['y' for i in range(len(labels))]
    if boxes_list.has_field('scores'):
        cmap = plt.get_cmap('viridis')
        colors = [cmap(s.cpu().detach().item()) for s in boxes_list.get_field('scores')]

    fig,ax = plt.subplots(1, figsize=(10,10))
    img = img.clip(0, 1)
    # Display the image
    if bgr:
        ax.imshow(img[:,:,::-1])
    else:
        ax.imshow(img)
    for idx_box, box in enumerate(box_tensor):
        x, y, w, h = box.cpu().detach().tolist()
        patch = patches.Rectangle((x,y),w-x,h-y,linewidth=2,edgecolor=colors[idx_box],facecolor='none')
        if not boxes_list.has_field('scores') or boxes_list.get_field('scores')[idx_box] >= threshold:
            ax.add_patch(patch)
            ax.text(x, y-5, labels[idx_box].item(), c='y')

    plt.show()

def plot_img_boxes(images, boxes, index, cfg, backend=None, threshold=0.0):
    plot_single_img_boxes(images.tensors[index], boxes[index], cfg, backend=backend, threshold=threshold)

def plot_img_only(image, cfg):
    img = (image.permute(1, 2, 0).cpu().numpy() * cfg.INPUT.PIXEL_STD +
           cfg.INPUT.PIXEL_MEAN) / 255
    fig, ax = plt.subplots(1, figsize=(10, 10))
    img = img.clip(0, 1)
    ax.imshow(img[:, :, ::-1])

    plt.show()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout