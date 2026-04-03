import warnings


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", ".*Warning only once for all operators.*")
    from . import bmm_outer_product  # noqa: F401
