import argparse

from segish.expand_annotations import load_and_expand

parser = argparse.ArgumentParser(description='Expand annotations based on similar pixels.')
parser.add_argument('image_path', type=str, help='Path to RGB image')
parser.add_argument('annotation_paths', type=str, nargs='+',
                    help='Paths to images with class annotations')
parser.add_argument('--window_size', type=int, default=3,
                    help='Size of window to consider as nearby pixels \
                            must be odd and >= 3')

args = parser.parse_args()
if args.window_size % 2 != 1 or args.window_size < 3:
    raise AttributeError("Invalid window size. Must be odd and >= 3.")

load_and_expand(args.image_path, args.annotation_paths, args.window_size)
