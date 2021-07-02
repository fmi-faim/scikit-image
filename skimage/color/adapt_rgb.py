import functools
import inspect

import numpy as np

from .. import color
from .._shared.utils import slice_at_axis
from ..util.dtype import _convert


__all__ = ['adapt_rgb', 'hsv_value', 'each_channel']


def is_rgb_like(image, rgb_axis=-1):
    """Return True if the image *looks* like it's RGB.

    This function should not be public because it is only intended to be used
    for functions that don't accept volumes as input, since checking an image's
    shape is fragile.
    """
    if rgb_axis is None:
        return False
    return (image.ndim == 3) and (image.shape[rgb_axis] in (3, 4))


def adapt_rgb(apply_to_rgb):
    """Return decorator that adapts to RGB images to a gray-scale filter.

    This function is only intended to be used for functions that don't accept
    volumes as input, since checking an image's shape is fragile.

    Parameters
    ----------
    apply_to_rgb : function
        Function that returns a filtered image from an image-filter and RGB
        image. This will only be called if the image is RGB-like.
    """
    def decorator(image_filter):
        @functools.wraps(image_filter)
        def image_filter_adapted(image, *args, **kwargs):
            rgb_axis = kwargs.get('rgb_axis', -1)
            if is_rgb_like(image, rgb_axis=rgb_axis):
                return apply_to_rgb(image_filter, image, *args, **kwargs)
            else:
                return image_filter(image, *args, **kwargs)
        return image_filter_adapted
    return decorator


def hsv_value(image_filter, image, *args, rgb_axis=-1, **kwargs):
    """Return color image by applying `image_filter` on HSV-value of `image`.

    Note that this function is intended for use with `adapt_rgb`.

    Parameters
    ----------
    image_filter : function
        Function that filters a gray-scale image.
    image : array
        Input image. Note that RGBA images are treated as RGB.
    rgb_axis : int or None, optional
        This parameter specifies which axis corresponds to `channels`.
    """
    # Slice the first three channels so that we remove any alpha channels.
    rgb_axis = rgb_axis % image.ndim
    image = image[slice_at_axis(slice(3), axis=rgb_axis)]
    hsv = color.rgb2hsv(image, channel_axis=rgb_axis)
    v_slice = slice_at_axis(2, axis=rgb_axis)
    value = hsv[v_slice].copy()
    value = image_filter(value, *args, **kwargs)
    hsv[v_slice] = _convert(value, hsv.dtype)
    return color.hsv2rgb(hsv, channel_axis=rgb_axis)


def each_channel(image_filter, image, *args, rgb_axis=-1, **kwargs):
    """Return color image by applying `image_filter` on channels of `image`.

    Note that this function is intended for use with `adapt_rgb`.

    Parameters
    ----------
    image_filter : function
        Function that filters a gray-scale image.
    image : array
        Input image.
    rgb_axis : int or None, optional
        This parameter specifies which axis corresponds to `channels`.
    """
    c_new = [image_filter(c, *args, **kwargs)
             for c in np.moveaxis(image, source=rgb_axis, destination=0)]
    return np.stack(c_new, axis=rgb_axis)
