## @package visualize
# Module caffe2.python.visualize
"""Functions that could be used to visualize Tensors.

This is adapted from the old-time iceberk package that Yangqing wrote... Oh gold
memories. Before decaf and caffe. Why iceberk? Because I was at Berkeley,
bears are vegetarian, and iceberg lettuce has layers of leaves.

(This joke is so lame.)
"""

import numpy as np
from matplotlib import cm, pyplot


def ChannelFirst(arr):
    """Convert a HWC array to CHW."""
    ndim = arr.ndim
    return arr.swapaxes(ndim - 1, ndim - 2).swapaxes(ndim - 2, ndim - 3)


def ChannelLast(arr):
    """Convert a CHW array to HWC."""
    ndim = arr.ndim
    return arr.swapaxes(ndim - 3, ndim - 2).swapaxes(ndim - 2, ndim - 1)


class PatchVisualizer(object):
    """PatchVisualizer visualizes patches.
  """

    def __init__(self, gap=1):
        self.gap = gap

    def ShowSingle(self, patch, cmap=None):
        """Visualizes one single patch.

    The input patch could be a vector (in which case we try to infer the shape
    of the patch), a 2-D matrix, or a 3-D matrix whose 3rd dimension has 3
    channels.
    """
        if len(patch.shape) == 1:
            patch = patch.reshape(self.get_patch_shape(patch))
        elif len(patch.shape) > 2 and patch.shape[2] != 3:
            raise ValueError("The input patch shape isn't correct.")
        # determine color
        if len(patch.shape) == 2 and cmap is None:
            cmap = cm.gray
        pyplot.imshow(patch, cmap=cmap)
        return patch

    def ShowMultiple(self, patches, ncols=None, cmap=None, bg_func=np.mean):
        """Visualize multiple patches.

    In the passed in patches matrix, each row is a patch, in the shape of either
    n*n, n*n*1 or n*n*3, either in a flattened format (so patches would be a
    2-D array), or a multi-dimensional tensor. We will try our best to figure
    out automatically the patch size.
    """
        num_patches = patches.shape[0]
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(num_patches)))
        nrows = int(np.ceil(num_patches / float(ncols)))
        if len(patches.shape) == 2:
            patches = patches.reshape(
                (patches.shape[0], ) + self.get_patch_shape(patches[0])
            )
        patch_size_expand = np.array(patches.shape[1:3]) + self.gap
        image_size = patch_size_expand * np.array([nrows, ncols]) - self.gap
        if len(patches.shape) == 4:
            if patches.shape[3] == 1:
                # gray patches
                patches = patches.reshape(patches.shape[:-1])
                image_shape = tuple(image_size)
                if cmap is None:
                    cmap = cm.gray
            elif patches.shape[3] == 3:
                # color patches
                image_shape = tuple(image_size) + (3, )
            else:
                raise ValueError("The input patch shape isn't expected.")
        else:
            image_shape = tuple(image_size)
            if cmap is None:
                cmap = cm.gray
        image = np.ones(image_shape) * bg_func(patches)
        for pid in range(num_patches):
            row = pid // ncols * patch_size_expand[0]
            col = pid % ncols * patch_size_expand[1]
            image[row:row+patches.shape[1], col:col+patches.shape[2]] = \
                patches[pid]
        pyplot.imshow(image, cmap=cmap, interpolation='nearest')
        pyplot.axis('off')
        return image

    def ShowImages(self, patches, *args, **kwargs):
        """Similar to ShowMultiple, but always normalize the values between 0 and 1
    for better visualization of image-type data.
    """
        patches = patches - np.min(patches)
        patches /= np.max(patches) + np.finfo(np.float64).eps
        return self.ShowMultiple(patches, *args, **kwargs)

    def ShowChannels(self, patch, cmap=None, bg_func=np.mean):
        """ This function shows the channels of a patch.

    The incoming patch should have shape [w, h, num_channels], and each channel
    will be visualized as a separate gray patch.
    """
        if len(patch.shape) != 3:
            raise ValueError("The input patch shape isn't correct.")
        patch_reordered = np.swapaxes(patch.T, 1, 2)
        return self.ShowMultiple(patch_reordered, cmap=cmap, bg_func=bg_func)

    def get_patch_shape(self, patch):
        """Gets the shape of a single patch.

    Basically it tries to interpret the patch as a square, and also check if it
    is in color (3 channels)
    """
        edgeLen = np.sqrt(patch.size)
        if edgeLen != np.floor(edgeLen):
            # we are given color patches
            edgeLen = np.sqrt(patch.size / 3.)
            if edgeLen != np.floor(edgeLen):
                raise ValueError("I can't figure out the patch shape.")
            return (edgeLen, edgeLen, 3)
        else:
            edgeLen = int(edgeLen)
            return (edgeLen, edgeLen)


_default_visualizer = PatchVisualizer()
"""Utility functions that directly point to functions in the default visualizer.

These functions don't return anything, so you won't see annoying printouts of
the visualized images. If you want to save the images for example, you should
explicitly instantiate a patch visualizer, and call those functions.
"""


class NHWC(object):
    @staticmethod
    def ShowSingle(*args, **kwargs):
        _default_visualizer.ShowSingle(*args, **kwargs)

    @staticmethod
    def ShowMultiple(*args, **kwargs):
        _default_visualizer.ShowMultiple(*args, **kwargs)

    @staticmethod
    def ShowImages(*args, **kwargs):
        _default_visualizer.ShowImages(*args, **kwargs)

    @staticmethod
    def ShowChannels(*args, **kwargs):
        _default_visualizer.ShowChannels(*args, **kwargs)


class NCHW(object):
    @staticmethod
    def ShowSingle(patch, *args, **kwargs):
        _default_visualizer.ShowSingle(ChannelLast(patch), *args, **kwargs)

    @staticmethod
    def ShowMultiple(patch, *args, **kwargs):
        _default_visualizer.ShowMultiple(ChannelLast(patch), *args, **kwargs)

    @staticmethod
    def ShowImages(patch, *args, **kwargs):
        _default_visualizer.ShowImages(ChannelLast(patch), *args, **kwargs)

    @staticmethod
    def ShowChannels(patch, *args, **kwargs):
        _default_visualizer.ShowChannels(ChannelLast(patch), *args, **kwargs)
