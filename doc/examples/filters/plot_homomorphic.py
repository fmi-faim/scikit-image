import functools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from skimage import data, filters


eagle = data.eagle()
cutoff = 0.02

# Create a Butterworth highpass filter
highpass1 = functools.partial(
    filters.butterworth,
    cutoff_frequency_ratio=cutoff,
    true_butterworth=True,
    npad=64,
)

eagle_highpass1 = highpass1(eagle, cutoff_frequency_ratio=cutoff)

amplitude_range = [0.3, 1]

# Create a second highpass filter that has a lower range > 0 so that low
# frequencies are not entirely suppressed.
highpass2 = functools.partial(
    filters.butterworth,
    cutoff_frequency_ratio=cutoff,
    true_butterworth=True,
    amplitude_range=amplitude_range,
    npad=64,
)

eagle_highpass2 = highpass2(eagle, cutoff_frequency_ratio=cutoff)


def homomorphic(highpass_function, image, eps=0.001):
    """ Apply homomorphic filtering using a given highpass filtering function.
    """
    output = image + eps
    np.log(output, out=output)
    output = highpass_function(output)
    np.exp(output, out=output)
    output -= eps
    return output


eagle_homomorphic1 = homomorphic(highpass1, eagle)
eagle_homomorphic2 = homomorphic(highpass2, eagle)

gridspec_kw = dict(wspace=0.05, hspace=0.2)
fig, axes = plt.subplots(2, 3, figsize=(11, 8), gridspec_kw=gridspec_kw)

title_font = dict(fontsize=12, fontweight='bold')

percentiles = [1, 99]
vmin, vmax = np.percentile(eagle, q=percentiles)

axes[0, 0].imshow(eagle, vmin=vmin, vmax=vmax, cmap='gray')
axes[0, 0].set_title('original', fontdict=title_font)

vmin, vmax = np.percentile(eagle_highpass1, q=percentiles)
axes[0, 1].imshow(eagle_highpass1, vmin=vmin, vmax=vmax, cmap='gray')
axes[0, 1].set_title('highpass\namplitude_range=[0, 1]',
                     fontdict=title_font)

vmin, vmax = np.percentile(eagle_highpass2, q=percentiles)
axes[0, 2].imshow(eagle_highpass2, vmin=vmin, vmax=vmax, cmap='gray')
axes[0, 2].set_title(f'highpass\n(amplitude_range={amplitude_range}',
                     fontdict=title_font)

axes[1, 0].set_axis_off()

vmin, vmax = np.percentile(eagle_homomorphic1, q=percentiles)
axes[1, 1].imshow(eagle_homomorphic1, vmin=vmin, vmax=vmax, cmap='gray')
axes[1, 1].set_title('homomorphic\namplitude_range=[0, 1]',
                     fontdict=title_font)

vmin, vmax = np.percentile(eagle_homomorphic2, q=percentiles)
axes[1, 2].imshow(eagle_homomorphic2, vmin=vmin, vmax=vmax, cmap='gray')
axes[1, 2].set_title(f'homomorphic\namplitude_range={amplitude_range}',
                     fontdict=title_font)

# remove tickmarks from all subplots
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

# plt.tight_layout()
plt.show()
