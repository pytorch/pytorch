
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

class _BaseQuantizedModule(object):
    @classmethod
    def from_float(cls, mod):
        if not hasattr(cls, '_FLOAT_MODULE'):
            raise NotImplementedError(
                "Class {} doesn't have _FLOAT_MODULE".format(cls.__name__))
        assert type(mod) == cls._FLOAT_MODULE, \
            str(cls) + \
            ".from_float expects an instance of " + \
            str(cls._FLOAT_MODULE)
        assert hasattr(mod, 'observer'), \
            type(mod) + " doesn't have an observer"
        scale, zero_point = mod.observer.calculate_qparams()[:2]
        new_mod = cls()
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod
