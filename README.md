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


## Example

![Initial segmentation scribbles](https://github.com/RossMcKenzie/segish/blob/main/examples/segish_initial.png)
These are the initial classification scribbles overlaying the base image.

![Expanded segmentation based on optimisation](https://github.com/RossMcKenzie/segish/blob/main/examples/segish_expanded.png)
The expanded classifications from Segish. 

## Limitations

There are still many pixel level errors that would need to be edited by hand. There is also an unsegmented area around the edge due to the current implementation using no padding (planned to be fixed in v1.1).

Much of the image, however, is correctly segmented which will save labelling time.

