# import cucim
from warnings import warn

import numpy as np

import skimage.morphology as _skimage_morphology

try:
    import diplib as dip
    have_diplib = True
    ndi_mode_translation_dict = dict(
        constant='add zeros',
        nearest='zero order',
        mirror='mirror',
        wrap='periodic')

except ImportError:
    have_diplib = False
    ndi_mode_translation_dict = {}
    numpy_mode_translation_dict = {}


def _to_diplib_mode(mode, cval=0):
    """Convert from skimage mode name to the corresponding ndimage mode."""
    if mode not in ndi_mode_translation_dict:
        # warnings.warn(f"diplib does not support mode {mode}")
        return NotImplemented

    if mode == 'constant' and cval != 0.:
        # warnings.warn(f"diplib backend only supports cval=0 for 'constant' mode")
        return NotImplemented

    return ndi_mode_translation_dict[mode]


# Backend support for skimage.filters

__ua_domain__ = 'numpy.skimage.morphology'
_implemented = {}


def __ua_convert__(dispatchables, coerce):
    if coerce:
        try:
            replaced = [
                np.asarray(d.value) if d.coercible and d.type is np.ndarray
                else d.value for d in dispatchables]
        except TypeError:
            return NotImplemented
    else:
        replaced = [d.value for d in dispatchables]

    if not all(isinstance(r, np.ndarray) for r in replaced):
        return NotImplemented

    return replaced


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    print(f"fn={fn}")
    if fn is None:
        print(f"_implemented={_implemented}, method={method}")
        return NotImplemented
    return fn(*args, **kwargs)


def _implements(skimage_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[skimage_func] = func
        return func
    return inner


@_implements(_skimage_morphology.binary_erosion)
def binary_erosion(image, footprint=None, out=None):

    if not have_diplib:
        raise ImportError("PyDIP (DIPlib) is unavailable.")

    image = np.asarray(image, dtype=bool)
    selem = np.asarray(footprint, dtype=bool)
    if np.all(selem):
        # dip::Image has opposite dimension order to NumPy arrays
        # so have to reverse the shape
        selem = diplib.SE(selem.shape[::-1], 'rectangular')
    else:
        selem = diplib.SE(diplib.Image(selem, None))

    if out is None:
        out = np.empty_like(image)
    elif out.dtype not in [np.float32, np.float64]:
        raise ValueError(
            "Provided out's dtype must be np.float32 or np.float64."
        )

    out[...] = diplib.Erosion(
        image,
        se=selem,
        boundaryCondition=["add max"] * image.ndim)
    return output


@_implements(_skimage_morphology.binary_dilation)
def binary_dilation(image, footprint=None, out=None):

    if not have_diplib:
        raise ImportError("PyDIP (DIPlib) is unavailable.")

    image = np.asarray(image, dtype=bool)
    selem = np.asarray(footprint, dtype=bool)
    if np.all(selem):
        # dip::Image has opposite dimension order to NumPy arrays
        # so have to reverse the shape
        selem = diplib.SE(selem.shape[::-1], 'rectangular')
    else:
        selem = diplib.SE(diplib.Image(selem, None))

    if out is None:
        out = np.empty_like(image)
    elif out.dtype not in [np.float32, np.float64]:
        raise ValueError(
            "Provided out's dtype must be np.float32 or np.float64."
        )

    out[...] = diplib.Dilation(
        image,
        se=selem,
        boundaryCondition=["add min"] * image.ndim)
    return output
