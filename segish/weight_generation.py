import numpy as np
from scipy import sparse


def rolling_window(image, win_size):
    """Creates a new ndarray made of a rolling square window.
    No padding is used the edge pixels are cut.

    Args:
        image (ndarray): 2D array to roll window over
        win_size (int): Size of window

    Returns:
        ndarray: 4D Array where element (i,j) is a size x size window
        centred on element (i, j) in image.

    """
    if win_size % 2 != 1 or win_size < 3:
        raise AttributeError("Invalid window size. Must be odd and >= 3.")
    s = (image.shape[0] - win_size + 1,) + (image.shape[1] - win_size + 1,)
    s += (win_size, win_size)
    strides = image.strides + image.strides
    return np.lib.stride_tricks.as_strided(image, shape=s, strides=strides)


def expand_to_window(image, win_size):
    """Creates a new 4D ndarray by copying elements of a 2D array.

    Args:
        image (ndarray): 2D array to copy.
        win_size (int): Size of window.

    Returns:
        ndarray: 4D Array where element (i,j) is a size x size array
        where every element equals element (i, j) in image.

    """
    image_new = image[..., np.newaxis, np.newaxis]
    image_new = np.tile(image_new, (1, 1, win_size, win_size))
    return image_new


def index_window(shape, win_size):
    """Tracks the flattened index of pixels in a rolling window array.

    Args:
        shape (tuple[int]): The size of the original array that is being windowed.
        win_size (int): Size of window.

    Returns:
        ndarray: 4D Array where element (i,j) is a size x size window
        centred on the index of (i, j).

    """
    erode = int(win_size / 2)
    num = shape[0] * shape[1]
    idx = np.arange(num)
    idx = np.reshape(idx, shape)
    idx_a = expand_to_window(idx[erode:-erode, erode:-erode], win_size)
    idx_b = rolling_window(idx, win_size)
    return idx_a, idx_b


def get_mean_std(image, win_size):
    """Finds the mean and standard deviation of rolling windows across a 2D array.

    Args:
        image (ndarray): 2D array to roll window over
        win_size (int): Size of window

    Returns:
        ndarray: 4D Array where element (i,j) is the mean of a size x size window
        centred on element (i, j) in image.
        ndarray: 4D Array where element (i,j) is the standard deviation of a
        size x size window centred on element (i, j) in image.

    """
    windows = rolling_window(image, win_size)
    mean = np.mean(windows, axis=(2, 3))
    std = np.std(windows, axis=(2, 3))
    return mean, std


def get_w_2D(image, win_size):
    """Finds the squared difference between the values in nearby pixels.

    Args:
        image (ndarray): A single channel from an image.
        win_size (int): Size of window for nearby pixels

    Returns:
        ndarray: 4D array where element (i,j) is a win_size x win_size array of
        the differences between element (i,j) from image
        and the pixels neighbouring it.

    """
    erode = int(win_size/2)
    _, std = get_mean_std(image, win_size)
    windows = rolling_window(image, win_size)
    
    yr = expand_to_window(image[erode:-erode, erode:-erode], win_size)
    yrs = (yr - windows) ** 2
    # if yrs is 0 std dev will also be 0 so ignore 0/0 errors and set results to 0
    with np.errstate(invalid='ignore'):
        exp = -yrs / expand_to_window(2 * (std ** 2), win_size)
    exp[np.isnan(exp)] = 0
    w = np.exp(exp)
    w[:, :, erode, erode] = 0
    return w


def get_w_inter(image, win_size):
    """Finds the mean squared difference between the values in nearby pixels
    across all channels in an image.

    Args:
        image (ndarray): A multi-channel image.
        win_size (int): Size of window for nearby pixels

    Returns:
        ndarray: 4D array where element (i,j) is a win_size x win_size array of
        the differences between element (i,j) from image
        and the pixels near it.

    """
    erode = int(win_size/2)
    w3D = []
    for i in range(image.shape[-1]):
        w3D.append(get_w_2D(image[..., i], win_size))
    w3d = np.stack(w3D, axis=2)
    w = np.mean(w3d, axis=2)
    sums = np.sum(w, axis=(2, 3))
    w = w / expand_to_window(sums, win_size)
    w = -w
    w[:, :, erode, erode] = 1
    return w


def get_sparse_weights(image, ann, win_size=3):
    """Finds the mean squared difference between the values in nearby pixels
    across all channels in an image. Puts them in a sparse 2D array.
    Sets annotated pixels as constraints.

    Args:
        image (ndarray): A multi-channel image.
        ann (ndarray): Multi-channel annotations
        win_size (int): Size of window for nearby pixels

    Returns:
        csc_matrix: Sparse 2D array where element (i,j) is the difference
        between pixel i and pixel j.
    """
    # Get weights and associated idxs
    mid = int(win_size / 2)
    w = get_w_inter(image, win_size) 
    
    seg_idx = get_ann_in_w(ann, win_size)
    w[seg_idx] = 0
    
    idx_a, idx_b = index_window(image.shape[:2], win_size)
    idx_a = idx_a.flatten()
    idx_b = idx_b.flatten()
    
    # Remove all diagonals when flattening w as they will be filled in later
    w[..., mid, mid] = np.nan
    w = w.flatten()
    idx_a = idx_a[~np.isnan(w)]
    idx_b = idx_b[~np.isnan(w)]
    w = w[~np.isnan(w)]
    
    # Fill in all diagonals including those not contained in W
    size = image.shape[0] * image.shape[1]
    diags_idx = np.arange(size)
    diags_v = np.ones(size)
    
    w = np.concatenate([w, diags_v], axis=0)
    idx_a = np.concatenate([idx_a, diags_idx], axis=0)
    idx_b = np.concatenate([idx_b, diags_idx], axis=0)
    
    sparse_w = sparse.csc_matrix((w, (idx_a, idx_b)), (size, size))
    return sparse_w
    

def get_ann_in_w(ann, win_size):
    """Creates a boolean index for annotated pixels that 
    matches the size of a rolling window array.

    Args:
        ann (ndarray): Multi-channel annotations
        win_size (int): Size of window

    Returns:
        ndarray: Boolean array of annotated pixels.

    """
    erode = int(win_size/2)
    idx = np.sum(ann, axis=2) != 0
    idx = idx[erode:-erode, erode:-erode]
    return idx
        
def get_flat_ann_idx(ann):
    """Creates a flattened index for annotated pixels.

    Args:
        ann (ndarray): Multi-channel annotations

    Returns:
        ndarray: Array of annotated pixel indexes.

    """
    Y = ann[:, :, 0].flatten()
    return np.nonzero(Y)[0]
