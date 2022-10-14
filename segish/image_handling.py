from pathlib import Path

import imageio.v3 as iio
import numpy as np


def load_image_with_annotations(image_path, annotation_paths, down_scaling=1):
    """ Loads and normalises images and rough segmentation annotations.
    Assumes images as rgb uint8.
    Removes alpha channel if present.
    """
    img = iio.imread(image_path)
    if down_scaling != 1:
        img = img[::down_scaling, ::down_scaling]
    img = img / 255
    if img.shape[2] > 3:
        img = img[..., :2]

    ann_l = []
    for ann_path in annotation_paths:
        ann = iio.imread(ann_path)
        if down_scaling != 1:
            ann = ann[::down_scaling, ::down_scaling]
        if ann.shape[2] > 3:
            ann = ann[..., :2]

        ann = np.sum(ann, axis=2)
        # Image loading adds extra pixels which this removes
        ann[ann < ann.max() / 20] = 0.
        ann[ann > ann.max() / 20] = 1.
        ann_l.append(ann)
    
    anns = np.stack(ann_l, axis=2)

    return img, anns


def save_adjacent(annotation_paths, anns):
    """Saves new annotations in same folder as originals.
    Assumes segmentation values in range 0-1.
    """
    assert(len(annotation_paths) == anns.shape[2])

    out_paths = []
    for in_path in annotation_paths:
        path = Path(in_path)
        out_paths.append(path.with_name(f'{path.stem}_expanded{path.suffix}'))
    save_annotations(out_paths, anns)


def save_annotations(out_paths, anns):
    """Saves new annotations as seperate files.
    Assumes segmentation values in range 0-1.
    """
    assert(len(out_paths) == anns.shape[2])
    for i in range(len(out_paths)):
        iio.imwrite(out_paths[i], (anns[..., i] * 255).astype(np.uint8))
