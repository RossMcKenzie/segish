import numpy as np
from scipy.sparse.linalg import splu

from .image_handling import load_image_with_annotations, save_adjacent
from .weight_generation import get_sparse_weights


def load_and_expand(image_path, annotation_paths, window_size=3):
    """Expands rough annotations loaded from disk to fill likely object boundaries
    and saves results.

    Args:
        img_path (str or Path): Path to original image being segmented
        annotation_paths(list[str or Path]): Paths to rough class annotations
        saved as seperate images
        window_size (int): Sets the window size to consider when determining
        pixel colour. The defualt will work for most situations.

    """
    img, ann = load_image_with_annotations(image_path, annotation_paths)
    expanded_ann = expand_annotations(img, ann, window_size)
    save_adjacent(annotation_paths, expanded_ann)


def expand_annotations(image, annotations, window_size=3):
    """Expands rough annotations to fill likely object boundaries.

    Args:
        image (ndarray): Multichannel image
        annotations(ndarray): Multichannel annotations
        window_size (int): Sets the window size to consider when determining

    Returns:
        ndarray: Binary annotations saved as seperate channels


    """
    w = get_sparse_weights(image, annotations, window_size)
    expanded_annotations = solve_for_annotations_and_weights(annotations, w)
    expanded_annotations = argmax_annotations(expanded_annotations)
    return expanded_annotations


def solve_for_annotations_and_weights(annotations, w):
    """Finds optimal spread of annotations based on initial annotations
    and pixel similarities.

    Args:
        annotations(ndarray): Multichannel annotations
        w (scipy.sparse.csc_matrix): Sparse representations of nearby pixel similarity

    Returns:
        ndarray: New annotations based on optimisation of pixel similarities


    """
    solver = splu(w)
    expanded_annotations = []
    for i in range(annotations.shape[2]):
        solution = solver.solve(annotations[..., i].astype('float32').flatten())
        solution = solution.reshape(annotations.shape[:2])
        expanded_annotations.append(solution)
    expanded_annotations = np.stack(expanded_annotations, axis=2)
    return expanded_annotations


def argmax_annotations(annotations):
    """Selects thre most likely class for each pixel and sets the
    pixel to only that class.

    Args:
        annotations(ndarray): Multichannel annotations

    Returns:
        ndarray: Argmaxed annotations

     """
    most_likely_class = np.argmax(annotations, axis=2)
    m_anns = annotations.copy()
    for i in range(annotations.shape[2]):
        m_anns[..., i][most_likely_class != i] = 0
    m_anns[m_anns > 0] = 1
    return m_anns
