import functools

# from numpy import ndarray
from uarray import generate_multimethod, Dispatchable
from uarray import all_of_type, create_multimethod
from unumpy import dtype, ndarray, mark_dtype

import skimage.filters
from skimage.filters import _api


__all__ = ['gaussian', 'difference_of_gaussians', 'gabor', 'gabor_kernel',
           'median', 'rank_order', 'correlate_sparse', 'unsharp_mask',
           'window', 'sobel', 'sobel_h', 'sobel_v', 'scharr', 'scharr_h',
           'scharr_v', 'prewitt', 'prewitt_h', 'prewitt_v', 'roberts',
           'roberts_pos_diag', 'roberts_neg_diag', 'laplace', 'farid',
           'farid_h', 'farid_v', 'forward', 'inverse', 'wiener',
           'compute_hessian_eigenvalues', 'meijering', 'sato', 'frangi',
           'hessian', 'try_all_threshold', 'threshold_local', 'threshold_otsu',
           'threshold_yen', 'threshold_isodata', 'threshold_li',
           'threshold_minimum', 'threshold_mean', 'threshold_triangle',
           'threshold_niblack', 'threshold_sauvola',
           'apply_hysteresis_threshold', 'threshold_multiotsu']


create_skimage_filters = functools.partial(
    create_multimethod, domain="numpy.skimage.filters"
)


def _identity_arg_replacer(args, kwargs, arrays):
    return args, kwargs


def _image_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """
    def self_method(image, *args, **kwargs):
        return (dispatchables[0],) + args, kwargs
    return self_method(*args, **kwargs)


def _image_kernel_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """
    def self_method(image, kernel, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs
    return self_method(*args, **kwargs)


def _image_low_high_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """
    def self_method(image, low, high, *args, **kwargs):
        return tuple(dispatchables[:3]) + args, kwargs
    return self_method(*args, **kwargs)


def _dispatch_identity(func):
    """
    Function annotation that creates a uarray multimethod from the function
    """
    return generate_multimethod(func, _identity_arg_replacer, domain="numpy.skimage.filters")


""" _gaussian.py multimethods """

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=None, preserve_range=False, truncate=4.0, *,
             channel_axis=None):
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


""" _gabor.py multimethods """

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


""" _median.py multimethods """

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def median(image, selem=None, out=None, mode='nearest', cval=0.0,
           behavior='ndimage'):
    return (image,)
median.__doc__ = _api.median.__doc__


""" _rank_order.py multimethods """

@create_skimage_filters(_image_arg_replacer)
def rank_order(image):
    return
rank_order.__doc__ = _api.rank_order.__doc__


""" _sparse.py multimethods """

@create_skimage_filters(_image_kernel_arg_replacer)
@all_of_type(ndarray)
def correlate_sparse(image, kernel, mode='reflect'):
    return (image,)
correlate_sparse.__doc__ = _api.correlate_sparse.__doc__

from .._shared import utils


""" _unsharp_mask.py multimethods """

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def unsharp_mask(image, radius=1.0, amount=1.0, multichannel=False,
                 preserve_range=False, *, channel_axis=None):
    return (image,)
unsharp_mask.__doc__ = _api.unsharp_mask.__doc__


""" _window.py multimethods """

@create_skimage_filters(_identity_arg_replacer)
def window(window_type, shape, warp_kwargs=None):
    return
window.__doc__ = _api.window.__doc__


""" edges.py multimethods """


def _image_mask_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """
    def self_method(image, mask, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kw_out
    return self_method(*args, **kwargs)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return (image, mask)
sobel.__doc__ = _api.sobel.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def sobel_h(image, mask=None):
    return (image, mask)
sobel_h.__doc__ = _api.sobel_h.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def sobel_v(image, mask=None):
    return (image, mask)
sobel_v.__doc__ = _api.sobel_v.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def scharr(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return (image, mask)
scharr.__doc__ = _api.scharr.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def scharr_h(image, mask=None):
    return (image, mask)
scharr_h.__doc__ = _api.scharr_h.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def scharr_v(image, mask=None):
    return (image, mask)
scharr_v.__doc__ = _api.scharr_v.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def prewitt(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return (image, mask)
prewitt.__doc__ = _api.prewitt.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def prewitt_h(image, mask=None):
    return (image, mask)
prewitt_h.__doc__ = _api.prewitt_h.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def prewitt_v(image, mask=None):
    return (image, mask)
prewitt_v.__doc__ = _api.prewitt_v.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def roberts(image, mask=None):
    return (image, mask)
roberts.__doc__ = _api.roberts.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def roberts_pos_diag(image, mask=None):
    return (image, mask)
roberts_pos_diag.__doc__ = _api.roberts_pos_diag.__doc__


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def roberts_neg_diag(image, mask=None):
    return (image, mask)
roberts_neg_diag.__doc__ = _api.roberts_neg_diag.__doc__


def _image_ksize_mask_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """
    def self_method(image, ksize, mask, *args, **kwargs):
        return (dispatchables[0], ksize, dispatchables[1]) + args, kw_out
    return self_method(*args, **kwargs)


@create_skimage_filters(_image_ksize_mask_arg_replacer)
@all_of_type(ndarray)
def laplace(image, ksize=3, mask=None):
    return (image, mask)
laplace.__doc__ = _api.laplace.__doc__


def _image_kwarg_mask_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """
    def self_method(image, *args, **kwargs):
        kwargs_out = kwargs.copy()
        kwargs_out["mask"] = dispatchables[1]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)


@create_skimage_filters(_image_kwarg_mask_arg_replacer)
@all_of_type(ndarray)
def farid(image, *, mask=None):
    return (image, mask)
farid.__doc__ = _api.farid.__doc__


@create_skimage_filters(_image_kwarg_mask_arg_replacer)
@all_of_type(ndarray)
def farid_h(image, *, mask=None):
    return (image, mask)
farid_h.__doc__ = _api.farid_h.__doc__


@create_skimage_filters(_image_kwarg_mask_arg_replacer)
@all_of_type(ndarray)
def farid_v(image, *, mask=None):
    return (image, mask)
farid_v.__doc__ = _api.farid_v.__doc__


""" lpi_filter.py multimethods """

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def forward(data, impulse_response=None, filter_params={},
            predefined_filter=None):
    return (data,)
forward.__doc__ = _api.forward.__doc__

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def inverse(data, impulse_response=None, filter_params={}, max_gain=2,
            predefined_filter=None):
    return (data,)
inverse.__doc__ = _api.inverse.__doc__

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def wiener(data, impulse_response=None, filter_params={}, K=0.25,
           predefined_filter=None):
    return (data,)
wiener.__doc__ = _api.wiener.__doc__


""" ridges.py multimethods """

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def compute_hessian_eigenvalues(image, sigma, sorting='none', mode='constant',
                                cval=0):
    return (image,)
compute_hessian_eigenvalues.__doc__ = _api.compute_hessian_eigenvalues.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def meijering(image, sigmas=range(1, 10, 2), alpha=None, black_ridges=True,
              mode='reflect', cval=0):
    return (image,)
meijering.__doc__ = _api.meijering.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def sato(image, sigmas=range(1, 10, 2), black_ridges=True, mode=None, cval=0):
    return (image,)
sato.__doc__ = _api.sato.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def frangi(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None,
           alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect',
           cval=0):
    return (image,)
frangi.__doc__ = _api.frangi.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def hessian(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None,
            alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode=None,
            cval=0):
    return (image,)
hessian.__doc__ = _api.hessian.__doc__


""" thresholding.py multimethods """

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def try_all_threshold(image, figsize=(8, 5), verbose=True):
    return (image,)
try_all_threshold.__doc__ = _api.try_all_threshold.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def threshold_local(image, block_size, method='gaussian', offset=0,
                    mode='reflect', param=None, cval=0):
    return (image,)
threshold_local.__doc__ = _api.threshold_local.__doc__


def _image_kwarg_hist_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """
    def self_method(image, *args, **kwargs):
        kw_out = kwargs.copy()
        if "hist" in kw:
            kw_out["hist"] = dispatchables[1]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)


@create_skimage_filters(_image_kwarg_hist_arg_replacer)
@all_of_type(ndarray)
def threshold_otsu(image=None, nbins=256, *, hist=None):
    return (image, hist)
threshold_otsu.__doc__ = _api.threshold_otsu.__doc__


@create_skimage_filters(_image_kwarg_hist_arg_replacer)
@all_of_type(ndarray)
def threshold_yen(image=None, nbins=256, *, hist=None):
    return (image, hist)
threshold_yen.__doc__ = _api.threshold_yen.__doc__


@create_skimage_filters(_image_kwarg_hist_arg_replacer)
@all_of_type(ndarray)
def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
    return (image, hist)
threshold_isodata.__doc__ = _api.threshold_isodata.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def threshold_li(image, *, tolerance=None, initial_guess=None,
                 iter_callback=None):
    return (image, )
threshold_li.__doc__ = _api.threshold_li.__doc__


@create_skimage_filters(_image_kwarg_hist_arg_replacer)
@all_of_type(ndarray)
def threshold_minimum(image=None, nbins=256, max_iter=10000, *, hist=None):
    return (image, hist)
threshold_minimum.__doc__ = _api.threshold_minimum.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def threshold_mean(image):
    return (image,)
threshold_mean.__doc__ = _api.threshold_mean.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def threshold_triangle(image, nbins=256):
    return (image,)
threshold_triangle.__doc__ = _api.threshold_triangle.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def threshold_niblack(image, window_size=15, k=0.2):
    return (image,)
threshold_niblack.__doc__ = _api.threshold_niblack.__doc__


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def threshold_sauvola(image, window_size=15, k=0.2, r=None):
    return (image,)
threshold_sauvola.__doc__ = _api.threshold_sauvola.__doc__


# TODO: low, high can be float or ndarray
@create_skimage_filters(_image_low_high_arg_replacer)
@all_of_type(ndarray)
def apply_hysteresis_threshold(image, low, high):
    return (image, low, high)
apply_hysteresis_threshold.__doc__ = _api.apply_hysteresis_threshold.__doc__


# TODO: low, high can be float or ndarray
@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def threshold_multiotsu(image, classes=3, nbins=256):
    return (image,)
threshold_multiotsu.__doc__ = _api.threshold_multiotsu.__doc__
