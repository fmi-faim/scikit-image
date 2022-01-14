"""Benchmarks for `skimage.morphology`.

See "Writing benchmarks" in the asv docs for more information.
"""

import numpy as np
from numpy.lib import NumpyVersion as Version

import skimage
from skimage import data, morphology, util


class Skeletonize3d(object):

    def setup(self, *args):
        try:
            # use a separate skeletonize_3d function on older scikit-image
            if Version(skimage.__version__) < Version('0.16.0'):
                self.skeletonize = morphology.skeletonize_3d
            else:
                self.skeletonize = morphology.skeletonize
        except AttributeError:
            raise NotImplementedError("3d skeletonize unavailable")

        # we stack the horse data 5 times to get an example volume
        self.image = np.stack(5 * [util.invert(data.horse())])

    def time_skeletonize_3d(self):
        self.skeletonize(self.image)

    def peakmem_reference(self, *args):
        """Provide reference for memory measurement with empty benchmark.

        Peakmem benchmarks measure the maximum amount of RAM used by a
        function. However, this maximum also includes the memory used
        during the setup routine (as of asv 0.2.1; see [1]_).
        Measuring an empty peakmem function might allow us to disambiguate
        between the memory used by setup and the memory used by target (see
        other ``peakmem_`` functions below).

        References
        ----------
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    def peakmem_skeletonize_3d(self):
        self.skeletonize(self.image)

class Reconstruction(object):

    def setup(self, *args):
        self.reconstruction = morphology.reconstruction

        x = np.linspace(0, 4 * np.pi)
        y_mask = np.cos(x)

        y_seed = y_mask.min() * np.ones_like(x)
        y_seed[0] = 0.5
        y_seed[-1] = 0

        self.seed = y_seed
        self.mask = y_mask

    def time_reconstruction(self):
        self.reconstruction(self.seed, self.mask)

    def peakmem_reference(self, *args):
        pass

    def peakmem_reconstruction(self):
        self.reconstruction(self.seed, self.mask)
