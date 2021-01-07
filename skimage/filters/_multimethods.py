import functools

# from numpy import ndarray
from uarray import generate_multimethod, Dispatchable
from uarray import all_of_type, create_multimethod
from unumpy import dtype, ndarray, mark_dtype, mark_non_coercible

import skimage.filters
from skimage.filters import _api

# , mark_as
#mark_dtype = mark_as(dtype)
#mark_non_coercible = lambda x: Dispatchable(x, ndarray, coercible=False)

create_skimage_filters = functools.partial(create_multimethod, domain="numpy.skimage.filters")


# def _dtype_argreplacer(args, kwargs, dispatchables):
#     def replacer(*a, dtype=None, **kw):
#         out_kw = kw.copy()
#         out_kw["dtype"] = dispatchables[0]
#         if "out" in out_kw:
#             out_kw["out"] = dispatchables[1]

#         return a, out_kw

#     return replacer(*args, **kwargs)


# def _self_argreplacer(args, kwargs, dispatchables):
#     def self_method(a, *args, **kwargs):
#         kw_out = kwargs.copy()
#         if "out" in kw_out:
#             kw_out["out"] = dispatchables[1]

#         return (dispatchables[0],) + args, kw_out

#     return self_method(*args, **kwargs)


def _identity_arg_replacer(args, kwargs, arrays):
    return args, kwargs


def _image_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the transform input array (``x``)
    """
    if len(args) > 0:
        return (dispatchables[0],) + args[1:], kwargs
    kw = kwargs.copy()
    kw['image'] = dispatchables[0]
    return args, kw



def _dispatch_identity(func):
    """
    Function annotation that creates a uarray multimethod from the function
    """
    return generate_multimethod(func, _identity_arg_replacer, domain="numpy.skimage.filters")


"""
def _dispatch_image(func):
    return generate_multimethod(func, _image_arg_replacer, domain="numpy.skimage.filters")

@_dispatch_image
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=None, preserve_range=False, truncate=4.0):
    return (Dispatchable(image, np.ndarray),)
"""

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=None, preserve_range=False, truncate=4.0):
    return (image,)
gaussian.__doc__ = _api.gaussian.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def difference_of_gaussians(image, low_sigma, high_sigma=None, *,
                            mode='nearest', cval=0, multichannel=False,
                            truncate=4.0):
    return (image,)
difference_of_gaussians.__doc__ = _api.difference_of_gaussians.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def _guess_spatial_dimensions(image):
    return (image,)
_guess_spatial_dimensions.__doc__ = _api._guess_spatial_dimensions.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def gabor(image, frequency, theta=0, bandwidth=1, sigma_x=None,
          sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0):
    return (image, )
gabor.__doc__ = _api.gabor.__doc__


@create_skimage_filters(_identity_arg_replacer)
def gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
                 n_stds=3, offset=0):
    return
gabor_kernel.__doc__ = _api.gabor_kernel.__doc__
