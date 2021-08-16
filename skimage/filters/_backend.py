import uarray as ua
from skimage.filters import _api


class _ScikitImageFiltersBackend:
    """The default backend for filters calculations

    Notes
    -----
    We use the domain ``numpy.skimage`` rather than ``skimage`` because in the
    future, ``uarray`` will treat the domain as a hierarchy. This means the user
    can install a single backend for ``numpy`` and have it implement
    ``numpy.skimage.filters`` as well.
    """
    __ua_domain__ = "numpy.skimage.filters"

    @staticmethod
    def __ua_function__(method, args, kwargs):
        fn = getattr(_api, method.__name__, None)

        if fn is None:
            return NotImplemented
        return fn(*args, **kwargs)


_named_backends = {
    'skimage.filters': _ScikitImageFiltersBackend,
}


def _backend_from_arg(backend):
    """Maps strings to known backends and validates the backend"""

    if isinstance(backend, str):
        try:
            backend = _named_backends[backend]
        except KeyError as e:
            raise ValueError('Unknown backend {}'.format(backend)) from e

    if backend.__ua_domain__ != 'numpy.skimage.filters':
        raise ValueError('Backend does not implement "numpy.skimage.filters"')

    return backend


def set_global_backend(backend, coerce=False, only=False, try_last=False):
    """Sets the global filters backend

    The global backend has higher priority than registered backends, but lower
    priority than context-specific backends set with `set_backend`.

    Parameters
    ----------
    backend: {object, 'skimage.filters'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'skimage.filters'} or an object that implements the uarray protocol.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    try_last : bool
        If ``True``, the global backend is tried after registered backends.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.skimage.filters``.

    Notes
    -----
    This will overwrite the previously set global backend, which, by default, is
    the SciPy implementation.

    Examples
    --------
    We can set the global skimage.filters backend:

    >>> import numpy as np
    >>> from skimage import filters
    >>> from skimage.filters import set_global_backend
    >>> set_global_backend("skimage")  # Sets global backend. "skimage" is the default backend.
    >>> filters.gaussian(np.array([1., 2., 2., 2., 1.]))  # Calls the global backend
    array([1.30039443, 1.69490604, 1.88288636, 1.69490604, 1.30039443])
    """
    backend = _backend_from_arg(backend)
    ua.set_global_backend(backend, coerce=coerce, only=only, try_last=try_last)


def register_backend(backend):
    """
    Register a backend for permanent use.

    Registered backends have the lowest priority and will be tried after the
    global backend.

    Parameters
    ----------
    backend: {object, 'skimage.filters'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'skimage.filters'} or an object that implements the uarray protocol.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.skimage.filters``.

    Examples
    --------
    We can register a new filters backend:

    >>> from skimage import filters
    >>> from skimage.filters import register_backend, set_global_backend
    >>> class NoopBackend:  # Define an invalid Backend
    ...     __ua_domain__ = "numpy.skimage.filters"
    ...     def __ua_function__(self, func, args, kwargs):
    ...          return NotImplemented
    >>> set_global_backend(NoopBackend())  # Set the invalid backend as global
    >>> register_backend("skimage")  # Register a new backend
    >>> filters.gaussian(np.array([1., 2., 2., 2., 1.]))  # The registered backend is called because the global backend returns `NotImplemented`
    array([1.30039443, 1.69490604, 1.88288636, 1.69490604, 1.30039443])
    >>> set_global_backend("skimage")  # Restore global backend to default

    """ # noqa
    backend = _backend_from_arg(backend)
    ua.register_backend(backend)


def set_backend(backend, coerce=False, only=False):
    """Context manager to set the backend within a fixed scope.

    Upon entering the ``with`` statement, the given backend will be added to
    the list of available backends with the highest priority. Upon exit, the
    backend is reset to the state before entering the scope.

    Parameters
    ----------
    backend: {object, 'skimage.filters'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'skimage.filters'} or an object that implements the uarray protocol.
    coerce: bool, optional
        Whether to allow expensive conversions for the ``x`` parameter. e.g.,
        copying a NumPy array to the GPU for a CuPy backend. Implies ``only``.
    only: bool, optional
       If only is ``True`` and this backend returns ``NotImplemented``, then a
       BackendNotImplemented error will be raised immediately. Ignoring any
       lower priority backends.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import filters
    >>> with filters.set_backend('skimage', only=True):
    ...     filters.gaussian(np.array([1., 2., 2., 2., 1.]))  # Always calls the skimage implementation

    """
    backend = _backend_from_arg(backend)
    return ua.set_backend(backend, coerce=coerce, only=only)


def skip_backend(backend):
    """Context manager to skip a backend within a fixed scope.

    Within the context of a ``with`` statement, the given backend will not be
    called. This covers backends registered both locally and globally. Upon
    exit, the backend will again be considered.

    Parameters
    ----------
    backend: {object, 'skimage.filters'}
        The backend to skip.
        Can either be a ``str`` containing the name of a known backend
        {'skimage.filters'} or an object that implements the uarray protocol.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import filters
    >>> x = np.array([1., 2., 2., 2., 1.])
    >>> filters.gaussian(x)  # Calls default SciPy backend
    array([1.30039443, 1.69490604, 1.88288636, 1.69490604, 1.30039443])
    >>> with filters.skip_backend('skimage'):  # We explicitly skip the SciPy backend
    ...     filters.gaussian(x)                # leaving no implementation available
    Traceback (most recent call last):
        ...
    BackendNotImplementedError: No selected backends had an implementation ...
    """
    backend = _backend_from_arg(backend)
    return ua.skip_backend(backend)


set_global_backend('skimage.filters', coerce=True, try_last=True)
