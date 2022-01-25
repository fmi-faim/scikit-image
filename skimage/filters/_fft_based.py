import functools

import numpy as np
import scipy.fft as fft

from .._shared.utils import _supported_float_type


def _get_ND_butterworth_filter(shape, factor, order, high_pass, real,
                               dtype=np.float64, true_butterworth=False,
                               amplitude_range=(0.0, 1.0)):
    """Create a N-dimensional Butterworth mask for an FFT

    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image
    true_butterworth : bool, optional
        When False, the square of the Butterworth filter is used.
    amplitude_range : 2-tuple of float, optional
        The Frequency response will have amplitudes in the specified range.

    Returns
    -------
    wfilt : ndarray
        The FFT mask.

    """
    ranges = []
    for i, d in enumerate(shape):
        # start and stop ensures center of mask aligns with center of FFT
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        ranges.append(fft.ifftshift(axis ** 2))
    # for real image FFT, halve the last axis
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    # q2 = squared Euclidian distance grid
    q2 = functools.reduce(
            np.add, np.meshgrid(*ranges, indexing="ij", sparse=True)
            )
    q2 = q2.astype(dtype)
    wfilt = 1 / (1 + np.power(q2, order))
    if high_pass:
        wfilt = 1 - wfilt
    if true_butterworth:
        np.sqrt(wfilt, out=wfilt)

    # rescale to desired amplitude range
    low, high = amplitude_range
    if low < 0 or high < low:
        raise ValueError(
            "Expected 0 <= amplitude_range[0] <= amplitude_range[1].")
    if not (low == 0.0 and high == 1.0):
        # adjust range to [low, high]
        wfilt = low + wfilt * (high - low)
    return wfilt


def butterworth(
    image,
    cutoff_frequency_ratio=0.005,
    high_pass=True,
    order=2.0,
    channel_axis=None,
    *,
    true_butterworth=False,
    amplitude_range=(0.0, 1.0),
    npad=0,
):
    """Apply a Butterworth filter to enhance high or low frequency features.

    This filter is defined in the Fourier domain.

    Parameters
    ----------
    image : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    cutoff_frequency_ratio : float, optional
        Determines the position of the cut-off relative to the shape of the
        FFT. This should be in the range [0, 0.5].
    high_pass : bool, optional
        Whether to perform a high pass filter. If False, a low pass filter is
        performed.
    order : float, optional
        Order of the filter which affects the slope near the cut-off. Higher
        order means steeper slope in frequency space.
    channel_axis : int, optional
        If there is a channel dimension, provide the index here. If None
        (default) then all axes are assumed to be spatial dimensions.
    true_butterworth : bool, optional
        When False, the square of a true Butterworth filter is used.
    amplitude_range : 2-tuple of float, optional
        The Frequency response will have amplitudes in the specified range.
    npad : int, optional
        Pad each edge of the by `npad` pixels using `numpy.pad`'s
        ``mode='edge'`` extension. Increase this if boundary artifacts are
        apparent.

    Returns
    -------
    result : ndarray
        The Butterworth-filtered image.

    Notes
    -----
    A band-pass filter can be achieved by combining a high pass and low
    pass filter.

    The literature contains multiple conventions for the functional form of
    the Butterworth filter. Here, with the default ``true_butterworth=False``
    it is implemented as the n-dimensional form of the squared Butterworth
    which in the lowpass case is given by

    .. math::
        H(f) = \\frac{1}{1 + \\left(\\frac{f}{c f_s}\\right)^{2n}}

    and in the highpass case by

    .. math::
        H(f) = \\frac{1}{1 + \\left(\\frac{c f_s}{f}\\right)^{2n}}

    with :math:`f=\\sqrt{\\sum_{d=0}^{\\mathrm{ndim}} f_{d}}` the absolute
    value of the spatial frequency, :math:`f_s` is the sampling frequency,
    :math:`c` the ``cutoff_frequency_ratio``, and :math:`n` the ``order``
    modeled after [2]_.

    If ``true_butterworth=True``, the filter is given by the square root of the
    above expression. This corresponds to the standard Butterworth as defined
    in the signal processing literature [3]_, [4]_.

    Note that `cutoff_frequency_ratio` is defined in terms of the sampling
    frequency, :math:`f_s`. The FFT spectrum covers the range
    :math:`[-f_s/2, f_s/2]` so `cutoff_frequency_ratio` should have a value
    between 0 and 0.5.

    When an `amplitude_range`, :math:`(a_{min}, a_{max})` is specified,
    a modified Butterworth, :math:`H^{\\prime} (f)` is used:

    .. math::
        H^{\\prime} (f) = a_{min}  + H(f) * (a_{max} - a_{min})

    which has amplitudes in the range :math:`[a_{min}, a_{max}]`


    Examples
    --------
    Apply a high pass and low pass Butterworth filter to a grayscale and
    color image respectively:

    >>> from skimage.data import camera, astronaut
    >>> from skimage.filters import butterworth
    >>> high_pass = butterworth(camera(), 0.07, True, 8)
    >>> low_pass = butterworth(astronaut(), 0.01, False, 4, channel_axis=-1)

    References
    ----------
    .. [1] Butterworth, Stephen. "On the theory of filter amplifiers."
           Wireless Engineer 7.6 (1930): 536-541.
    .. [2] Russ, John C., et al. The Image Processing Handbook, 3rd. Ed.
           1999, CRC Press, LLC.
    .. [3] https://en.wikipedia.org/wiki/Butterworth_filter
    .. [4] Birchfield, Stan. Image Processing and Analysis. 2018. Cengage
           Learning.

    """
    if npad < 0:
        raise ValueError("npad must be >= 0")
    elif npad > 0:
        center_slice = tuple(slice(npad, s + npad) for s in image.shape)
        image = np.pad(image, npad, mode='symmetric')
    fft_shape = (image.shape if channel_axis is None
                 else np.delete(image.shape, channel_axis))
    is_real = np.isrealobj(image)
    float_dtype = _supported_float_type(image.dtype, allow_complex=True)
    if cutoff_frequency_ratio < 0 or cutoff_frequency_ratio > 0.5:
        raise ValueError(
            "cutoff_frequency_ratio should be in the range [0, 0.5]"
        )

    if len(amplitude_range) != 2:
        raise ValueError("amplitude_range must be a pair of values")
    wfilt = _get_ND_butterworth_filter(
        fft_shape, cutoff_frequency_ratio, order, high_pass, is_real,
        float_dtype, true_butterworth, amplitude_range
    )
    axes = np.arange(image.ndim)
    if channel_axis is not None:
        axes = np.delete(axes, channel_axis)
        abs_channel = channel_axis % image.ndim
        post = image.ndim - abs_channel - 1
        sl = ((slice(None),) * abs_channel + (np.newaxis,) +
              (slice(None),) * post)
        wfilt = wfilt[sl]
    if is_real:
        butterfilt = fft.irfftn(wfilt * fft.rfftn(image, axes=axes),
                                s=fft_shape, axes=axes)
    else:
        butterfilt = fft.ifftn(wfilt * fft.fftn(image, axes=axes),
                               s=fft_shape, axes=axes)
    if npad > 0:
        butterfilt = butterfilt[center_slice]
    return butterfilt
