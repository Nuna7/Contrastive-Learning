# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
from task_adapter.utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import io
from .automatic_mask_generator import SeemAutomaticMaskGenerator
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

def interactive_seem_m2m_auto(model, image, text_size, label_mode='1', alpha=0.1, anno_mode=['Mask']):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    mask_generator = SeemAutomaticMaskGenerator(model)
    outputs = mask_generator.generate(images)

    from task_adapter.utils.visualizer import Visualizer
    visual = Visualizer(image_ori, metadata=metadata)
    sorted_anns = sorted(outputs, key=(lambda x: x['area']), reverse=False)
    label = 1

    mask_map = np.zeros(image_ori.shape, dtype=np.uint8) 

    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation'] # [0,1] bool matrix. mask.shape (img_width, img_height)

        if i == 0:
            mask_accum = mask
        else:
            mask = mask & ~mask_accum ## exclude previous regions to avoid overlapping
            mask_accum = mask_accum | mask ## then logic OR to accumulate current mask into previous masks
        if np.sum(mask) < 110: # do not annotate small regions
            continue

        # Draw annotations for im (with masks and labels)
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)

        # assign the mask to the mask_map
        mask_map[mask == 1] = label
        label += 1

    # Get the original im (already has masks and labels)
    im = demo.get_image()  
    
    # Create result_image by copying im and adding mask boundaries
    result_image_array = np.asarray(im).copy()
    
    # Add mask boundaries to result_image
    label = 1
    # Generate a set of distinct colors for boundaries
    np.random.seed(42)  # For consistent colors
    boundary_colors = np.random.randint(100, 255, (len(sorted_anns), 3), dtype=np.uint8)
    
    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        
        if i == 0:
            mask_accum = mask
        else:
            mask = mask & ~mask_accum
            mask_accum = mask_accum | mask
        if np.sum(mask) < 110:
            continue
            
        # Draw contours (boundaries) with different colors
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Use bright, distinct colors for each mask boundary
        boundary_color = boundary_colors[i % len(boundary_colors)]
        
        # Draw a double-line border for better visibility
        cv2.drawContours(result_image_array, contours, -1, (0, 0, 0), thickness=4)  # Black outline
        cv2.drawContours(result_image_array, contours, -1, tuple(boundary_color.tolist()), thickness=2)  # Colored inner line
        
        label += 1
    
    result_image = Image.fromarray(result_image_array.astype(np.uint8))

    return im, sorted_anns, mask_map, result_image

    # from task_adapter.utils.visualizer import Visualizer
    # visual = Visualizer(image_ori, metadata=metadata)
    # sorted_anns = sorted(outputs, key=(lambda x: x['area']), reverse=True)
    # label = 1
    # for ann in sorted_anns:
    #     mask = ann['segmentation']
    #     color_mask = np.random.random((1, 3)).tolist()[0]
    #     # color_mask = [int(c*255) for c in color_mask]
    #     demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
    #     label += 1
    # im = demo.get_image()

    # # fig=plt.figure(figsize=(10, 10))
    # # plt.imshow(image_ori)
    # # show_anns(outputs)
    # # fig.canvas.draw()
    # # im=Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    # return im


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))