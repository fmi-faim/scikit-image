from ._multimethods import (gabor, gabor_kernel)
from ._multimethods import (gaussian, _guess_spatial_dimensions,
                            difference_of_gaussians)
from ._multimethods import median
from ._multimethods import rank_order
from ._multimethods import correlate_sparse
from ._multimethods import unsharp_mask
from ._multimethods import window
from ._multimethods import (sobel, sobel_h, sobel_v,
                            scharr, scharr_h, scharr_v,
                            prewitt, prewitt_h, prewitt_v,
                            roberts, roberts_pos_diag, roberts_neg_diag,
                            laplace,
                            farid, farid_h, farid_v)
from ._multimethods import forward, inverse, wiener
from ._multimethods import meijering, sato, frangi, hessian
from ._multimethods import (threshold_local, threshold_otsu, threshold_yen,
                            threshold_isodata, threshold_li, threshold_minimum,
                            threshold_mean, threshold_triangle,
                            threshold_niblack, threshold_sauvola,
                            threshold_multiotsu, try_all_threshold,
                            apply_hysteresis_threshold)
from .lpi_filter import LPIFilter2D
from . import rank
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
