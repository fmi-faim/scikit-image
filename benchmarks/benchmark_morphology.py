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

    param_names = ['ndim']
    params = [1, 2, 3]

    def setup(self, ndim, *args):
        self.reconstruction = morphology.reconstruction

        if ndim == 1:
            n = 200000
        elif ndim == 2:
            n = 1000
        elif ndim == 3:
            n = 100
        else:
            n = 10
        coords = np.meshgrid(*((np.linspace(0, 20, n),) * ndim), sparse=True)
        bumps = 0
        for c in coords:
            bumps = bumps + np.sin(c)
        h = 0.3
        seed = bumps - h
        self.seed = seed
        self.mask = bumps

    def time_reconstruction(self, *args):
        self.reconstruction(self.seed, self.mask)

    def peakmem_reference(self, ndim, *args):
        pass

    def peakmem_reconstruction(self, *args):
        self.reconstruction(self.seed, self.mask)
