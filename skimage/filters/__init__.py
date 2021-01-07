from ._multimethods import gabor_kernel, gabor
from ._multimethods import (gaussian, _guess_spatial_dimensions,
                            difference_of_gaussians)
from .lpi_filter import inverse, wiener, LPIFilter2D
from .edges import (sobel, sobel_h, sobel_v,
                    scharr, scharr_h, scharr_v,
                    prewitt, prewitt_h, prewitt_v,
                    roberts, roberts_pos_diag, roberts_neg_diag,
                    laplace,
                    farid, farid_h, farid_v)
from ._rank_order import rank_order
from .thresholding import (threshold_local, threshold_otsu, threshold_yen,
                           threshold_isodata, threshold_li, threshold_minimum,
                           threshold_mean, threshold_triangle,
                           threshold_niblack, threshold_sauvola,
                           threshold_multiotsu, try_all_threshold,
                           apply_hysteresis_threshold)
from .ridges import (meijering, sato, frangi, hessian)
from . import rank
from ._median import median
from ._sparse import correlate_sparse
from ._unsharp_mask import unsharp_mask
from ._window import window
from ._backend import (set_backend, skip_backend, set_global_backend,
                       register_backend)


__all__ = ['inverse',
           'correlate_sparse',
           'wiener',
           'LPIFilter2D',
           'gaussian',
           'difference_of_gaussians',
           'median',
           'sobel',
           'sobel_h',
           'sobel_v',
           'scharr',
           'scharr_h',
           'scharr_v',
           'prewitt',
           'prewitt_h',
           'prewitt_v',
           'roberts',
           'roberts_pos_diag',
           'roberts_neg_diag',
           'laplace',
           'farid',
           'farid_h',
           'farid_v',
           'rank_order',
           'gabor_kernel',
           'gabor',
           'try_all_threshold',
           'meijering',
           'sato',
           'frangi',
           'hessian',
           'threshold_otsu',
           'threshold_yen',
           'threshold_isodata',
           'threshold_li',
           'threshold_local',
           'threshold_minimum',
           'threshold_mean',
           'threshold_niblack',
           'threshold_sauvola',
           'threshold_triangle',
           'threshold_multiotsu',
           'apply_hysteresis_threshold',
           'rank',
           'unsharp_mask',
           'window']
