# import cucim
import cupy as cp
import numpy as np
import skimage.filters as _skimage_filters
from cucim.skimage import filters as _cucim_filters

# Backend support for skimage.filters

__ua_domain__ = 'numpy.skimage'
_implemented = {}


def __ua_convert__(dispatchables, coerce):
    if coerce:
        try:
            replaced = [
                cp.asarray(d.value) if d.coercible and d.type is np.ndarray
                else d.value for d in dispatchables]
        except TypeError:
            return NotImplemented
    else:
        replaced = [d.value for d in dispatchables]

    if not all(d.type is not np.ndarray or isinstance(r, cp.ndarray)
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


@_implements(_skimage_filters.gaussian)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=False, preserve_range=False, truncate=4.0, *,
             channel_axis=None):
    if channel_axis is not None:
        if channel_axis == -1:
            multichannel=True
        else:
            return NotImplementedError(
                "TODO: add channel_axis support to cuCIM"
            )
    return _cucim_filters.gaussian(
        image, sigma=sigma, output=output, mode=mode, cval=cval,
        multichannel=multichannel, preserve_range=preserve_range,
        truncate=truncate)


gaussian.__doc__ = _cucim_filters.gaussian.__doc__
