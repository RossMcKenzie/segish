# Segish

A system for assisting segmentation of images using initial scribbles. Based on techniques from _Colorization using Optimization_, A. Levin et al., 2004.

This approach uses the similarity of the colour of neighbouring pixels to produce a fully segmented image based on some initial rough scribbles.

## Usage

Pip install `requirements.txt`

Segish can be installed as a package via `pip install .`

You can also use Segish as a script by passing the paths to your image and class annotations as separate images:

```console
python segish/expand_annotations.py image_path \
annotation_class_0_path [annotation_class_1_path ...]
```


