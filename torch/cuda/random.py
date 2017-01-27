from torch import _C
from . import _lazy_init


def get_rng_state():
    _lazy_init()
    return _C._cuda_getRNGState()


def set_rng_state(new_state):
    _lazy_init()
    return _C._cuda_setRNGState(new_state)


def manual_seed(seed):
    _lazy_init()
    return _C._cuda_manualSeed(seed)


def manual_seed_all(seed):
    _lazy_init()
    return _C._cuda_manualSeedAll(seed)


def seed():
    _lazy_init()
    return _C._cuda_seed()


def seed_all():
    _lazy_init()
    return _C._cuda_seedAll()


def initial_seed():
    _lazy_init()
    return _C._cuda_initialSeed()
