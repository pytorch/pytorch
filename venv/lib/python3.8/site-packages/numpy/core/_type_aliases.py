"""
Due to compatibility, numpy has a very large number of different naming
conventions for the scalar types (those subclassing from `numpy.generic`).
This file produces a convoluted set of dictionaries mapping names to types,
and sometimes other mappings too.

.. data:: allTypes
    A dictionary of names to types that will be exposed as attributes through
    ``np.core.numerictypes.*``

.. data:: sctypeDict
    Similar to `allTypes`, but maps a broader set of aliases to their types.

.. data:: sctypeNA
    NumArray-compatible names for the scalar types. Contains not only
    ``name: type`` mappings, but ``char: name`` mappings too.

    .. deprecated:: 1.16

.. data:: sctypes
    A dictionary keyed by a "type group" string, providing a list of types
    under that group.

"""
import warnings

from numpy.compat import unicode
from numpy._globals import VisibleDeprecationWarning
from numpy.core._string_helpers import english_lower, english_capitalize
from numpy.core.multiarray import typeinfo, dtype
from numpy.core._dtype import _kind_name


sctypeDict = {}      # Contains all leaf-node scalar types with aliases
class TypeNADict(dict):
    def __getitem__(self, key):
        # 2018-06-24, 1.16
        warnings.warn('sctypeNA and typeNA will be removed in v1.18 '
                      'of numpy', VisibleDeprecationWarning, stacklevel=2)
        return dict.__getitem__(self, key)
    def get(self, key, default=None):
        # 2018-06-24, 1.16
        warnings.warn('sctypeNA and typeNA will be removed in v1.18 '
                      'of numpy', VisibleDeprecationWarning, stacklevel=2)
        return dict.get(self, key, default)

sctypeNA = TypeNADict()  # Contails all leaf-node types -> numarray type equivalences
allTypes = {}            # Collect the types we will add to the module


# separate the actual type info from the abstract base classes
_abstract_types = {}
_concrete_typeinfo = {}
for k, v in typeinfo.items():
    # make all the keys lowercase too
    k = english_lower(k)
    if isinstance(v, type):
        _abstract_types[k] = v
    else:
        _concrete_typeinfo[k] = v

_concrete_types = {v.type for k, v in _concrete_typeinfo.items()}


def _bits_of(obj):
    try:
        info = next(v for v in _concrete_typeinfo.values() if v.type is obj)
    except StopIteration:
        if obj in _abstract_types.values():
            raise ValueError("Cannot count the bits of an abstract type")

        # some third-party type - make a best-guess
        return dtype(obj).itemsize * 8
    else:
        return info.bits


def bitname(obj):
    """Return a bit-width name for a given type object"""
    bits = _bits_of(obj)
    dt = dtype(obj)
    char = dt.kind
    base = _kind_name(dt)

    if base == 'object':
        bits = 0

    if bits != 0:
        char = "%s%d" % (char, bits // 8)

    return base, bits, char


def _add_types():
    for name, info in _concrete_typeinfo.items():
        # define C-name and insert typenum and typechar references also
        allTypes[name] = info.type
        sctypeDict[name] = info.type
        sctypeDict[info.char] = info.type
        sctypeDict[info.num] = info.type

    for name, cls in _abstract_types.items():
        allTypes[name] = cls
_add_types()

# This is the priority order used to assign the bit-sized NPY_INTxx names, which
# must match the order in npy_common.h in order for NPY_INTxx and np.intxx to be
# consistent.
# If two C types have the same size, then the earliest one in this list is used
# as the sized name.
_int_ctypes = ['long', 'longlong', 'int', 'short', 'byte']
_uint_ctypes = list('u' + t for t in _int_ctypes)

def _add_aliases():
    for name, info in _concrete_typeinfo.items():
        # these are handled by _add_integer_aliases
        if name in _int_ctypes or name in _uint_ctypes:
            continue

        # insert bit-width version for this class (if relevant)
        base, bit, char = bitname(info.type)

        myname = "%s%d" % (base, bit)

        # ensure that (c)longdouble does not overwrite the aliases assigned to
        # (c)double
        if name in ('longdouble', 'clongdouble') and myname in allTypes:
            continue

        base_capitalize = english_capitalize(base)
        if base == 'complex':
            na_name = '%s%d' % (base_capitalize, bit//2)
        elif base == 'bool':
            na_name = base_capitalize
        else:
            na_name = "%s%d" % (base_capitalize, bit)

        allTypes[myname] = info.type

        # add mapping for both the bit name and the numarray name
        sctypeDict[myname] = info.type
        sctypeDict[na_name] = info.type

        # add forward, reverse, and string mapping to numarray
        sctypeNA[na_name] = info.type
        sctypeNA[info.type] = na_name
        sctypeNA[info.char] = na_name

        sctypeDict[char] = info.type
        sctypeNA[char] = na_name
_add_aliases()

def _add_integer_aliases():
    seen_bits = set()
    for i_ctype, u_ctype in zip(_int_ctypes, _uint_ctypes):
        i_info = _concrete_typeinfo[i_ctype]
        u_info = _concrete_typeinfo[u_ctype]
        bits = i_info.bits  # same for both

        for info, charname, intname, Intname in [
                (i_info,'i%d' % (bits//8,), 'int%d' % bits, 'Int%d' % bits),
                (u_info,'u%d' % (bits//8,), 'uint%d' % bits, 'UInt%d' % bits)]:
            if bits not in seen_bits:
                # sometimes two different types have the same number of bits
                # if so, the one iterated over first takes precedence
                allTypes[intname] = info.type
                sctypeDict[intname] = info.type
                sctypeDict[Intname] = info.type
                sctypeDict[charname] = info.type
                sctypeNA[Intname] = info.type
                sctypeNA[charname] = info.type
            sctypeNA[info.type] = Intname
            sctypeNA[info.char] = Intname

        seen_bits.add(bits)

_add_integer_aliases()

# We use these later
void = allTypes['void']

#
# Rework the Python names (so that float and complex and int are consistent
#                            with Python usage)
#
def _set_up_aliases():
    type_pairs = [('complex_', 'cdouble'),
                  ('int0', 'intp'),
                  ('uint0', 'uintp'),
                  ('single', 'float'),
                  ('csingle', 'cfloat'),
                  ('singlecomplex', 'cfloat'),
                  ('float_', 'double'),
                  ('intc', 'int'),
                  ('uintc', 'uint'),
                  ('int_', 'long'),
                  ('uint', 'ulong'),
                  ('cfloat', 'cdouble'),
                  ('longfloat', 'longdouble'),
                  ('clongfloat', 'clongdouble'),
                  ('longcomplex', 'clongdouble'),
                  ('bool_', 'bool'),
                  ('bytes_', 'string'),
                  ('string_', 'string'),
                  ('str_', 'unicode'),
                  ('unicode_', 'unicode'),
                  ('object_', 'object')]
    for alias, t in type_pairs:
        allTypes[alias] = allTypes[t]
        sctypeDict[alias] = sctypeDict[t]
    # Remove aliases overriding python types and modules
    to_remove = ['ulong', 'object', 'int', 'float',
                 'complex', 'bool', 'string', 'datetime', 'timedelta',
                 'bytes', 'str']

    for t in to_remove:
        try:
            del allTypes[t]
            del sctypeDict[t]
        except KeyError:
            pass
_set_up_aliases()


sctypes = {'int': [],
           'uint':[],
           'float':[],
           'complex':[],
           'others':[bool, object, bytes, unicode, void]}

def _add_array_type(typename, bits):
    try:
        t = allTypes['%s%d' % (typename, bits)]
    except KeyError:
        pass
    else:
        sctypes[typename].append(t)

def _set_array_types():
    ibytes = [1, 2, 4, 8, 16, 32, 64]
    fbytes = [2, 4, 8, 10, 12, 16, 32, 64]
    for bytes in ibytes:
        bits = 8*bytes
        _add_array_type('int', bits)
        _add_array_type('uint', bits)
    for bytes in fbytes:
        bits = 8*bytes
        _add_array_type('float', bits)
        _add_array_type('complex', 2*bits)
    _gi = dtype('p')
    if _gi.type not in sctypes['int']:
        indx = 0
        sz = _gi.itemsize
        _lst = sctypes['int']
        while (indx < len(_lst) and sz >= _lst[indx](0).itemsize):
            indx += 1
        sctypes['int'].insert(indx, _gi.type)
        sctypes['uint'].insert(indx, dtype('P').type)
_set_array_types()


# Add additional strings to the sctypeDict
_toadd = ['int', 'float', 'complex', 'bool', 'object',
          'str', 'bytes', ('a', 'bytes_')]

for name in _toadd:
    if isinstance(name, tuple):
        sctypeDict[name[0]] = allTypes[name[1]]
    else:
        sctypeDict[name] = allTypes['%s_' % name]

del _toadd, name
