import matplotlib.pyplot as plt
import numpy as np

from skimage import data, filters

image = data.camera()

# cutoff frequencies as a fraction of the maximum frequency
cutoffs = [.005, .02, .08, .16]


def get_filtered(image, cutoffs, true_butterworth=False, order=3.0, npad=0):
    """Lowpass and highpass butterworth filtering at all specified cutoffs.

    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    cutoffs : sequence of int
        Both lowpass and highpass filtering will be performed for each cutoff
        frequency in `cutoffs`.
    true_butterworth : bool, optional
        Whether the traditional Butterworth filter or its square is used.
    order : float, optional
        The order of the Butterworth filter

    Returns
    -------
    lowpass_filtered : list of ndarray
        List of images lowpass filtered at the frequencies in `cutoffs`.
    highpass_filtered : list of ndarray
        List of images highpass filtered at the frequencies in `cutoffs`.
    """

    lowpass_filtered = []
    highpass_filtered = []
    for cutoff in cutoffs:
        lowpass_filtered.append(
            filters.butterworth(
                image,
                cutoff_frequency_ratio=cutoff,
                order=order,
                high_pass=False,
                true_butterworth=true_butterworth,
                npad=npad,
            )
        )
        highpass_filtered.append(
            filters.butterworth(
                image,
                cutoff_frequency_ratio=cutoff,
                order=order,
                high_pass=True,
                true_butterworth=true_butterworth,
                npad=npad,
            )
        )
    return lowpass_filtered, highpass_filtered


def plot_filtered(lowpass_filtered, highpass_filtered, cutoffs):
    """Generate plots for paired lists of lowpass and highpass images."""
    fig, axes = plt.subplots(2, 1 + len(cutoffs), figsize=(15, 6))
    fontdict = dict(fontsize=14, fontweight='bold')

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('original', fontdict=fontdict)
    axes[1, 0].set_axis_off()

    for i, c in enumerate(cutoffs):
        axes[0, i + 1].imshow(lowpass_filtered[i], cmap='gray')
        axes[0, i + 1].set_title(f'lowpass, c={c}', fontdict=fontdict)
        axes[1, i + 1].imshow(highpass_filtered[i], cmap='gray')
        axes[1, i + 1].set_title(f'highpass, c={c}', fontdict=fontdict)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


# Perform filtering with the (squared) Butterworth filter at a range of
# cutoffs.
lowpasses, highpasses = get_filtered(image, cutoffs, true_butterworth=False)

fig, axes = plot_filtered(lowpasses, highpasses, cutoffs)
titledict = dict(fontsize=18, fontweight='bold')
fig.text(0.5, 0.95, '(squared) Butterworth filtering (order=3.0, npad=0)',
         fontdict=titledict, horizontalalignment='center')

# It can be seen that for lower values of the cutoff, there are artifacts near
# the edge of the images. This is due to the periodic nature of the DFT and can
# be reduced by applying some amount of padding to the edges prior to
# filtering so that there are not sharp eges in the periodic extension of the
# image. This can be done via the ``npad`` argument to ``butterworth``.

lowpasses, highpasses = get_filtered(image, cutoffs, true_butterworth=False,
                                     npad=64)

fig, axes = plot_filtered(lowpasses, highpasses, cutoffs)
fig.text(0.5, 0.95, '(squared) Butterworth filtering (order=3.0, npad=64)',
         fontdict=titledict, horizontalalignment='center')

# Note that with padding, the undesired shading at the image edges is
# substantially reduced.

# To use the traditional signal processing defintion of the Butterworth filter,
# set ``true_butterworth=True``. This variaant has an amplitude profile in the
# frequency domain that is the square root of the the default case with
# ``true_butterworth=False``. The overall behavior is qualitatively similar.

lowpasses, highpasses = get_filtered(image, cutoffs, true_butterworth=True,
                                     npad=64)

fig, axes = plot_filtered(lowpasses, highpasses, cutoffs)
fig.text(0.5, 0.95, 'Butterworth filtering (order=3.0, npad=64)',
         fontdict=titledict, horizontalalignment='center')

plt.show()
