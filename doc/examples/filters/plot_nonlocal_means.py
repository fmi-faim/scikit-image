"""
=================================================
Non-local means denoising for preserving textures
=================================================

In this example, we denoise a detail of the astronaut image using the non-local
means filter. The non-local means algorithm replaces the value of a pixel by an
average of a selection of other pixels values: small patches centered on the
other pixels are compared to the patch centered on the pixel of interest, and
the average is performed only for pixels that have patches close to the current
patch. As a result, this algorithm can restore well textures, that would be
blurred by other denoising algoritm.
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr


astro = img_as_float(data.astronaut())
astro = astro[30:180, 150:300]

sigma = 0.08
noisy = astro + sigma * np.random.standard_normal(astro.shape)
noisy = np.clip(noisy, 0, 1)

# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(noisy, multichannel=True))
print("estimated noise standard deviation = {}".format(sigma_est))

# slow algorithm
denoise = denoise_nl_means(noisy, 5, 6, h=1.15*sigma_est, multichannel=True,
                           fast_mode=False)

# slow algorithm, sigma provided
denoise2 = denoise_nl_means(noisy, 5, 6, h=0.8*sigma_est, multichannel=True,
                            sigma=sigma_est, fast_mode=False)

# fast algorithm
denoise_fast = denoise_nl_means(noisy, 5, 6, h=0.8*sigma_est,
                                multichannel=True, fast_mode=True)

# fast algorithm, sigma provided
denoise2_fast = denoise_nl_means(noisy, 5, 6, h=0.6*sigma_est,
                                 multichannel=True, sigma=0.8*sigma_est,
                                 fast_mode=True)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6),
                       sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('noisy')
ax[0, 1].imshow(denoise)
ax[0, 1].axis('off')
ax[0, 1].set_title('non-local means\n(slow)')
ax[0, 2].imshow(denoise2)
ax[0, 2].axis('off')
ax[0, 2].set_title('non-local means\n(slow, using $\sigma_{est}$)')
ax[1, 0].imshow(astro)
ax[1, 0].axis('off')
ax[1, 0].set_title('original\n(noise free)')
ax[1, 1].imshow(denoise_fast)
ax[1, 1].axis('off')
ax[1, 1].set_title('non-local means\n(fast)')
ax[1, 2].imshow(denoise2_fast)
ax[1, 2].axis('off')
ax[1, 2].set_title('non-local means\n(fast, using $\sigma_{est}$)')

fig.tight_layout()

# print PSNR metric for each case
psnr_noisy = compare_psnr(astro, noisy)
psnr = compare_psnr(astro, denoise.astype(astro.dtype))
psnr2 = compare_psnr(astro, denoise2.astype(astro.dtype))
psnr_fast = compare_psnr(astro, denoise_fast.astype(astro.dtype))
psnr2_fast = compare_psnr(astro, denoise2_fast.astype(astro.dtype))

print("PSNR (noisy) = {:0.2f}".format(psnr_noisy))
print("PSNR (slow) = {:0.2f}".format(psnr))
print("PSNR (slow, using sigma) = {:0.2f}".format(psnr2))
print("PSNR (fast) = {:0.2f}".format(psnr_fast))
print("PSNR (fast, using sigma) = {:0.2f}".format(psnr2_fast))

plt.show()
