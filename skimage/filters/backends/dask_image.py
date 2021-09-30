# import cucim
import math
import numbers

import numpy as np
import dask.array as da

from skimage import filters
from skimage.util import apply_parallel
from skimage._shared.utils import _supported_float_type

# Backend support for skimage.filters

__ua_domain__ = 'numpy.skimage.filters'
_implemented = {}


# break up into chunks of approximately 512**2 elements
def asdask(array):
    if isinstance(array, da.Array):
        return array
    chunks = (round((512 ** 2)**(1/array.ndim)), ) * array.ndim
    return da.asarray(array, chunks=chunks)


def __ua_convert__(dispatchables, coerce):
    if coerce:
        # coerce array inputs to dask arrays
        try:
            replaced = [
                asdask(d.value) if d.coercible and d.type is np.ndarray
                else d.value for d in dispatchables]
        except TypeError:
            return NotImplemented
        allowed_array_types = da.Array
    else:
        # np.ndarray inputs will result in np.ndarray outputs
        # da.Array inputs will result in a dask array output
        replaced = [d.value for d in dispatchables]
        allowed_array_types = (np.ndarray, da.Array)

    if not all(d.type is not np.ndarray or isinstance(r, allowed_array_types)
               for r, d in zip(replaced, dispatchables)):
        return NotImplemented

    return replaced


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented

    # may need warnings or errors related to API changes here
    #if 'multichannel' in kwargs and not _skimage_1_0:
    #    warnings.warn('The \'multichannel\' argument is not supported for scikit-image >= 1.0')
    return fn(*args, **kwargs)


def _implements(scipy_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[scipy_func] = func
        return func

    return inner


def _assequence(x):
    """Convert scalars to a sequence, otherwise pass through ``x`` unchanged"""
    if isinstance(x, Number):
        return (x,)
    return x


def _get_gaussian_sigmas(ndim, sigma):

    nsigmas = np.array(sigma)
    if nsigmas.ndim == 0:
        nsigmas = np.array(ndim * [nsigmas[()]])

    if nsigmas.ndim != 1:
        raise RuntimeError(
            "Must have a single sigma or a single sequence."
        )

    if ndim != len(nsigmas):
        raise RuntimeError(
            "Must have an equal number of sigmas to image dimensions."
        )

    if not issubclass(nsigmas.dtype.type, numbers.Real):
        raise TypeError("Must have real sigmas.")

    nsigmas = tuple(nsigmas)

    return nsigmas


def _get_gaussian_border(ndim, sigma, truncate):
    sigma = np.array(_get_gaussian_sigmas(ndim, sigma))

    if not isinstance(truncate, numbers.Real):
        raise TypeError("Must have a real truncate value.")

    half_shape = tuple(np.ceil(sigma * truncate).astype(int))

    return half_shape

def _insert(seq, channel_axis, value):
    seq = list(seq)
    seq.insert(channel_axis, value)
    return tuple(seq)


@_implements(filters.gaussian)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=False, preserve_range=False, truncate=4.0, *,
             channel_axis=None):

    if channel_axis is None and multichannel:
        channel_axis = -1
        multichannel = False
    ndim = image.ndim if channel_axis is None else image.ndim - 1
    sigma = _get_gaussian_sigmas(ndim, sigma)
    depth = _get_gaussian_border(ndim, sigma, truncate)

    # passing channel_axis to apply_parallel takes care of insertion of 0 depth
    # passing channel_axis to gaussian takes care of insertion of 0 in sigma
    # if channel_axis is not None:
    #     channel_axis = channel_axis % image.ndim
    #     depth = _insert(depth, channel_axis, 0)
    #     sigma = _insert(sigma, channel_axis, 0)

    # depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")
    dtype = _supported_float_type(image.dtype)

    if output is not None:
        raise ValueError("output is unsupported")

    # print(f"image.chunksize={image.chunksize}")

    # handled depth and sigma above, so set channel_axis to None
    return apply_parallel(
        filters.gaussian,
        image,
        depth=depth,
        mode='wrap' if mode == 'wrap' else 'none',
        extra_keywords=dict(sigma=sigma,
                            mode=mode,
                            cval=cval,
                            preserve_range=preserve_range,
                            truncate=truncate,
                            channel_axis=channel_axis),
        dtype=dtype,
        # can set channel_axis to None since depth and sigma already
        channel_axis=channel_axis,
        )


gaussian.__doc__ = filters.gaussian.__doc__


@_implements(filters.difference_of_gaussians)
def difference_of_gaussians(image, low_sigma, high_sigma=None, *,
                            mode='nearest', cval=0, channel_axis=None,
                            truncate=4.0, multichannel=False):

    if channel_axis is None and multichannel:
        channel_axis = -1
        multichannel = False
    ndim = image.ndim if channel_axis is None else image.ndim - 1
    low_sigma = _get_gaussian_sigmas(ndim, low_sigma)
    if high_sigma is not None:
        high_sigma = _get_gaussian_sigmas(ndim, high_sigma)
    depth = _get_gaussian_border(ndim, high_sigma, truncate)
    # if channel_axis is not None:
    #     # depth = list(depth)
    #     # depth[channel_axis] = 1
    #     channel_axis = channel_axis % image.ndim
    #     depth = _insert(depth, channel_axis, 0)
    #     low_sigma = _insert(low_sigma, channel_axis, 0)
    #     if high_sigma is not None:
    #         high_sigma = _insert(high_sigma, channel_axis, 0)

    # depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")
    dtype = _supported_float_type(image.dtype)

    # handled depth and sigma above, so set channel_axis to None
    return apply_parallel(
        filters.difference_of_gaussians,
        image,
        depth=depth,
        mode='wrap' if mode == 'wrap' else 'none',
        extra_keywords=dict(low_sigma=low_sigma,
                            high_sigma=high_sigma,
                            mode=mode,
                            cval=cval,
                            truncate=truncate,
                            channel_axis=channel_axis),
        dtype=dtype,
        channel_axis=channel_axis,
        )


difference_of_gaussians.__doc__ = filters.difference_of_gaussians.__doc__


@_implements(filters.median)
def median(image, footprint=None, out=None, mode='nearest', cval=0.0,
           behavior='ndimage'):
    depth = tuple(math.ceil(s / 2) for s in footprint.shape)
    dtype = _supported_float_type(image.dtype)
    footprint = np.asarray(footprint)  # footprint should not be a dask array

    if out is not None:
        return NotImplemented

    return apply_parallel(
        filters.median,
        image,
        depth=depth,
        mode='wrap' if mode == 'wrap' else 'none',
        extra_keywords=dict(footprint=footprint,
                            mode=mode,
                            cval=cval,
                            behavior=behavior),
        dtype=dtype,
        )


median.__doc__ = filters.median.__doc__
