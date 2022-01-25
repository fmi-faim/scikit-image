import matplotlib.pyplot as plt
from matplotlib import rcParams

from skimage import data, filters

image = data.camera()

# cutoff frequencies as a fraction of the maximum frequency
cutoffs = [.005, .02, .08, .16]


def get_filtered(image, cutoffs, true_butterworth=False, order=3.0):
    """Lowpass and highpass butterworth filtering at all specified cutoffs.

    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    cutoffs : sequence of int
        The cutoff frequencies to repeat the filtering at.
    true_butterworth : bool, optional
        Whether the Butterworth filter or its square is used.
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
                true_butterworth=true_butterworth
            )
        )
        highpass_filtered.append(
            filters.butterworth(
                image,
                cutoff_frequency_ratio=cutoff,
                order=order,
                high_pass=True,
                true_butterworth=true_butterworth
            )
        )
    return lowpass_filtered, highpass_filtered


def plot_filtered(lowpass_filtered, highpass_filtered, cutoffs):
    """Generate plots for a pair of sequences of filtered images."""
    fig, axes = plt.subplots(2, 1 + len(cutoffs), figsize=(15, 6))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('original')
    axes[1, 0].set_axis_off()

    for i in range(len(cutoffs)):
        axes[0, i + 1].imshow(lowpass_filtered[i])
        axes[0, i + 1].set_ylabel('lowpass')
        axes[1, i + 1].imshow(highpass_filtered[i])
        axes[1, i + 1].set_ylabel('highpass')
        axes[1, i + 1].set_xlabel(f'cutoff {cutoffs[i]}')

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


# adjust default Matplotlib settings for better visibility
rcParams['image.cmap'] = 'gray'
rcParams['font.size'] = 14
rcParams['font.weight'] = 'bold'

# Perform filtering with the squared Butterworth filter
lowpasses, highpasses = get_filtered(image, cutoffs, true_butterworth=False)

# Plot the result at the various cutoff frequencies
fig, axes = plot_filtered(lowpasses, highpasses, cutoffs)
fig.text(0.5, 0.95, 'Filtering with squared Butterworth filters (order=3.0)',
         fontdict=dict(fontsize=18), horizontalalignment='center')

# Perform filtering with the true Butterworth filter
lowpasses, highpasses = get_filtered(image, cutoffs, true_butterworth=True)

# Plot the result at the various cutoff frequencies. The cutoff for the true
# filter is more gradual than for the squared one, so the amount of blurring
# in the lowpass cases are less than with `true_butterworth=False` at the same
# cutoff and order.
fig, axes = plot_filtered(lowpasses, highpasses, cutoffs)
fig.text(0.5, 0.95, 'Filtering with true Butterworth filters (order=3.0)',
         fontdict=dict(fontsize=18), horizontalalignment='center')

plt.show()
