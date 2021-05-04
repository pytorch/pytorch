## @package schema
# Module caffe2.python.schema
"""
Defines a minimal set of data types that allow to represent datasets with
arbitrary nested structure, including objects of variable length, such as
maps and lists.

This defines a columnar storage format for such datasets on top of caffe2
tensors. In terms of capacity of representation, it can represent most of
the data types supported by Parquet, ORC, DWRF file formats.

See comments in operator_test/dataset_ops_test.py for an example and
walkthrough on how to use schema to store and iterate through a structured
in-memory dataset.
"""





import logging
import numpy as np
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.core import BlobReference
from collections import OrderedDict, namedtuple
from past.builtins import basestring
from future.utils import viewitems, viewkeys, viewvalues
from itertools import islice
from six import StringIO
from typing import Sequence

logger = logging.getLogger(__name__)

FIELD_SEPARATOR = ':'


def _join_field_name(prefix, suffix):
    if prefix and suffix:
        return '{}{}{}'.format(prefix, FIELD_SEPARATOR, suffix)
    elif prefix:
        return prefix
    elif suffix:
        return suffix
    else:
        return ''


def _normalize_field(field_or_type_or_blob, keep_blobs=True):
    """Clones/normalizes a field before adding it to a container."""
    if isinstance(field_or_type_or_blob, Field):
        return field_or_type_or_blob.clone(keep_blobs=keep_blobs)
    elif type(field_or_type_or_blob) in (type, np.dtype):
        return Scalar(dtype=field_or_type_or_blob)
    else:
        return Scalar(blob=field_or_type_or_blob)


FeatureSpec = namedtuple(
    'FeatureSpec',
    [
        'feature_type',
        'feature_names',
        'feature_ids',
        'feature_is_request_only',
        'desired_hash_size',
        'feature_to_index',
    ]
)

# pyre-fixme[16]: `FeatureSpec.__new__` has no attribute `__defaults__`
FeatureSpec.__new__.__defaults__ = (None, None, None, None, None, None)


class Metadata(
    namedtuple(
        'Metadata', ['categorical_limit', 'expected_value', 'feature_specs']
    )
):
    """Represents additional information associated with a scalar in schema.

    `categorical_limit` - for fields of integral type that are guaranteed to be
    non-negative it specifies the maximum possible value plus one. It's often
    used as a size of an embedding table.

    `expected_value` - anticipated average value of elements in the field.
    Usually makes sense for length fields of lists.

    `feature_specs` - information about the features that contained in this
    field. For example if field have more than 1 feature it can have list of
    feature names contained in this field."""
    __slots__: Sequence[str] = ()


# pyre-fixme[16]: `Metadata.__new__` has no attribute `__defaults__`
Metadata.__new__.__defaults__ = (None, None, None)


class Field(object):
    """Represents an abstract field type in a dataset.
    """

    __slots__: Sequence[str] = ("_parent", "_field_offsets")

    def __init__(self, children):
        """Derived classes must call this after their initialization."""
        self._parent = (None, 0)
        offset = 0
        self._field_offsets = []
        for child in children:
            self._field_offsets.append(offset)
            offset += len(child.field_names())
        self._field_offsets.append(offset)

    def clone_schema(self):
        return self.clone(keep_blobs=False)

    def field_names(self):
        """Return the children field names for this field."""
        raise NotImplementedError('Field is an abstract class.')

    def field_types(self):
        """Return the numpy.dtype for each of the children fields."""
        raise NotImplementedError('Field is an abstract class.')

    def field_metadata(self):
        """Return the Metadata for each of the children fields."""
        raise NotImplementedError('Field is an abstract class.')

    def field_blobs(self):
        """Return the list of blobs with contents for this Field.
        Values can either be all numpy.ndarray or BlobReference.
        If any of the fields doesn't have a blob, throws.
        """
        raise NotImplementedError('Field is an abstract class.')

    def all_scalars(self):
        """Return the list of all Scalar instances in the Field.
        The order is the same as for field_names() or field_blobs()"""
        raise NotImplementedError('Field is an abstract class.')

    def has_blobs(self):
        """Return True if every scalar of this field has blobs."""
        raise NotImplementedError('Field is an abstract class.')

    def clone(self, keep_blobs=True):
        """Clone this Field along with its children."""
        raise NotImplementedError('Field is an abstract class.')

    def _set_parent(self, parent, relative_id):
        self._parent = (parent, relative_id)

    def slice(self):
        """
        Returns a slice representing the range of field ids that belong to
        this field. This slice can be used to index a list of fields.

        E.g.:

        >>> s = Struct(
        >>>     ('a', Scalar()),
        >>>     ('b', Struct(
        >>>         ('b1', Scalar()),
        >>>         ('b2', Scalar()),
        >>>     )),
        >>>     ('c', Scalar()),
        >>> )
        >>> field_data = ['da', 'db1', 'db2', 'dc']
        >>> field_data[s.b.split()]
        ['db1', 'db2']
        """
        base_id = self._child_base_id()
        return slice(base_id, base_id + len(self.field_names()))

    def _child_base_id(self, child_index=None):
        """Get the base id of the given child"""
        p, i = self._parent
        pos = 0 if child_index is None else self._field_offsets[child_index]
        if p:
            pos += p._child_base_id(i)
        return pos

    def __eq__(self, other):
        """Equivalance of two schemas"""
        return (
            (self.field_names() == other.field_names()) and
            (self.field_types() == other.field_types()) and
            (self.field_metadata() == other.field_metadata())
        )

    def _pprint_impl(self, indent, str_buffer):
        raise NotImplementedError('Field is an abstract class.')

    def __repr__(self):
        str_buffer = StringIO()
        self._pprint_impl(0, str_buffer)
        contents = str_buffer.getvalue()
        str_buffer.close()
        return contents


class List(Field):
    """Represents a variable-length list.

    Values of a list can also be complex fields such as Lists and Structs.
    In addition to the fields exposed by its `values` field, a List exposes an
    additional `lengths` field, which will contain the size of each list under
    the parent domain.
    """

    __slots__: Sequence[str] = ("lengths", "_items")

    def __init__(self, values, lengths_blob=None):
        if isinstance(lengths_blob, Field):
            assert isinstance(lengths_blob, Scalar)
            self.lengths = _normalize_field(lengths_blob)
        else:
            self.lengths = Scalar(np.int32, lengths_blob)
        self._items = _normalize_field(values)
        self.lengths._set_parent(self, 0)
        self._items._set_parent(self, 1)
        super(List, self).__init__([self.lengths, self._items])

    def field_names(self):
        value_fields = self._items.field_names()
        return (
            ['lengths'] + [_join_field_name('values', v) for v in value_fields]
        )

    def field_types(self):
        return self.lengths.field_types() + self._items.field_types()

    def field_metadata(self):
        return self.lengths.field_metadata() + self._items.field_metadata()

    def field_blobs(self):
        return self.lengths.field_blobs() + self._items.field_blobs()

    def all_scalars(self):
        return self.lengths.all_scalars() + self._items.all_scalars()

    def has_blobs(self):
        return self.lengths.has_blobs() and self._items.has_blobs()

    def clone(self, keep_blobs=True):
        return type(self)(
            _normalize_field(self._items, keep_blobs=keep_blobs),
            _normalize_field(self.lengths, keep_blobs=keep_blobs)
        )

    def _pprint_impl(self, indent, str_buffer):
        str_buffer.write('  ' * indent + "List(\n")
        str_buffer.write('  ' * (indent + 1) + "lengths=\n")
        self.lengths._pprint_impl(indent=indent + 2, str_buffer=str_buffer)
        str_buffer.write('  ' * (indent + 1) + "_items=\n")
        self._items._pprint_impl(indent=indent + 2, str_buffer=str_buffer)
        str_buffer.write('  ' * indent + ")\n")

    def __getattr__(self, item):
        """If the value of this list is a struct,
        allow to introspect directly into its fields."""
        if item.startswith('__'):
            raise AttributeError(item)
        if isinstance(self._items, Struct):
            return getattr(self._items, item)
        elif item == 'value' or item == 'items':
            return self._items
        else:
            raise AttributeError('Field not found in list: %s.' % item)

    def __getitem__(self, item):
        names = item.split(FIELD_SEPARATOR, 1)

        if len(names) == 1:
            if item == 'lengths':
                return self.lengths
            elif item == 'values':
                return self._items
        else:
            if names[0] == 'values':
                return self._items[names[1]]
        raise KeyError('Field not found in list: %s.' % item)


class ListWithEvicted(List):
    """
    This class is similar with List, but containing extra field evicted_values for
    LRU Hashing.
    """

    __slots__: Sequence[str] = ("_evicted_values",)

    def __init__(self, values, lengths_blob=None, evicted_values=None):
        if isinstance(evicted_values, Field):
            assert isinstance(evicted_values, Scalar)
            self._evicted_values = _normalize_field(evicted_values)
        else:
            self._evicted_values = Scalar(np.int64, evicted_values)
        super(ListWithEvicted, self).__init__(values, lengths_blob=lengths_blob)

    def field_names(self):
        value_fields = self._items.field_names()
        return (
            ['lengths'] + [_join_field_name('values', v) for v in value_fields] + ["_evicted_values"]
        )

    def field_types(self):
        return self.lengths.field_types() + self._items.field_types() + self._evicted_values.field_types()

    def field_metadata(self):
        return self.lengths.field_metadata() + self._items.field_metadata() + self._evicted_values.field_metadata()

    def field_blobs(self):
        return self.lengths.field_blobs() + self._items.field_blobs() + self._evicted_values.field_blobs()

    def all_scalars(self):
        return self.lengths.all_scalars() + self._items.all_scalars() + self._evicted_values.all_scalars()

    def has_blobs(self):
        return self.lengths.has_blobs() and self._items.has_blobs() + self._evicted_values.has_blobs()

    def clone(self, keep_blobs=True):
        return type(self)(
            _normalize_field(self._items, keep_blobs=keep_blobs),
            _normalize_field(self.lengths, keep_blobs=keep_blobs),
            _normalize_field(self._evicted_values, keep_blobs=keep_blobs)
        )

    def _pprint_impl(self, indent, str_buffer):
        str_buffer.write('  ' * indent + "ListWithEvicted(\n")
        str_buffer.write('  ' * (indent + 1) + "lengths=\n")
        self.lengths._pprint_impl(indent=indent + 2, str_buffer=str_buffer)
        str_buffer.write('  ' * (indent + 1) + "_items=\n")
        self._items._pprint_impl(indent=indent + 2, str_buffer=str_buffer)
        str_buffer.write('  ' * (indent + 1) + "_evicted_values=\n")
        self._evicted_values._pprint_impl(indent=indent + 2, str_buffer=str_buffer)
        str_buffer.write('  ' * indent + ")\n")


    def __getattr__(self, item):
        """If the value of this list is a struct,
        allow to introspect directly into its fields."""
        if item.startswith('__'):
            raise AttributeError(item)
        if item == "_evicted_values":
            return self._evicted_values
        if isinstance(self._items, Struct):
            return getattr(self._items, item)
        elif item == 'value' or item == 'items':
            return self._items
        else:
            raise AttributeError('Field not found in list: %s.' % item)

    def __getitem__(self, item):
        names = item.split(FIELD_SEPARATOR, 1)

        if len(names) == 1:
            if item == 'lengths':
                return self.lengths
            elif item == 'values':
                return self._items
            elif item == '_evicted_values':
                return self._evicted_values
        else:
            if names[0] == 'values':
                return self._items[names[1]]
        raise KeyError('Field not found in list: %s.' % item)


class Struct(Field):
    """Represents a named list of fields sharing the same domain.
    """

    __slots__: Sequence[str] = ("fields", "_frozen")

    def __init__(self, *fields):
        """ fields is a list of tuples in format of (name, field). The name is
        a string of nested name, e.g., `a`, `a:b`, `a:b:c`. For example

        Struct(
          ('a', Scalar()),
          ('b:c', Scalar()),
          ('b:d:e', Scalar()),
          ('b', Struct(
            ('f', Scalar()),
          )),
        )

        is equal to

        Struct(
          ('a', Scalar()),
          ('b', Struct(
            ('c', Scalar()),
            ('d', Struct(('e', Scalar()))),
            ('f', Scalar()),
          )),
        )
        """
        for field in fields:
            assert len(field) == 2
            assert field[0], 'Field names cannot be empty'
            assert field[0] != 'lengths', (
                'Struct cannot contain a field named `lengths`.'
            )
        fields = [(name, _normalize_field(field)) for name, field in fields]
        self.fields = OrderedDict()
        for name, field in fields:
            if FIELD_SEPARATOR in name:
                name, field = self._struct_from_nested_name(name, field)
            if name not in self.fields:
                self.fields[name] = field
                continue
            if (
                    not isinstance(field, Struct) or
                    not isinstance(self.fields[name], Struct)
            ):
                raise ValueError('Duplicate field name: %s' % name)
            self.fields[name] = self.fields[name] + field
        for id, (_, field) in enumerate(viewitems(self.fields)):
            field._set_parent(self, id)
        super(Struct, self).__init__(viewvalues(self.fields))
        self._frozen = True

    def _struct_from_nested_name(self, nested_name, field):
        def create_internal(nested_name, field):
            names = nested_name.split(FIELD_SEPARATOR, 1)
            if len(names) == 1:
                added_field = field
            else:
                added_field = create_internal(names[1], field)
            return Struct((names[0], added_field))

        names = nested_name.split(FIELD_SEPARATOR, 1)
        assert len(names) >= 2
        return names[0], create_internal(names[1], field)

    def get_children(self):
        return list(viewitems(self.fields))

    def field_names(self):
        names = []
        for name, field in viewitems(self.fields):
            names += [_join_field_name(name, f) for f in field.field_names()]
        return names

    def field_types(self):
        types = []
        for _, field in viewitems(self.fields):
            types += field.field_types()
        return types

    def field_metadata(self):
        metadata = []
        for _, field in viewitems(self.fields):
            metadata += field.field_metadata()
        return metadata

    def field_blobs(self):
        blobs = []
        for _, field in viewitems(self.fields):
            blobs += field.field_blobs()
        return blobs

    def all_scalars(self):
        scalars = []
        for _, field in viewitems(self.fields):
            scalars += field.all_scalars()
        return scalars

    def has_blobs(self):
        return all(field.has_blobs() for field in viewvalues(self.fields))

    def clone(self, keep_blobs=True):
        normalized_fields = [
            (k, _normalize_field(v, keep_blobs=keep_blobs))
            for k, v in viewitems(self.fields)
        ]
        return type(self)(*normalized_fields)

    def _get_field_by_nested_name(self, nested_name):
        names = nested_name.split(FIELD_SEPARATOR, 1)
        field = self.fields.get(names[0], None)

        if field is None:
            return None

        if len(names) == 1:
            return field

        try:
            return field[names[1]]
        except (KeyError, TypeError):
            return None

    def _pprint_impl(self, indent, str_buffer):
        str_buffer.write('  ' * indent + "Struct( \n")
        for name, field in viewitems(self.fields):
            str_buffer.write('  ' * (indent + 1) + "{}=".format(name) + "\n")
            field._pprint_impl(indent=indent + 2, str_buffer=str_buffer)
        str_buffer.write('  ' * indent + ") \n")

    def __contains__(self, item):
        field = self._get_field_by_nested_name(item)
        return field is not None

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, item):
        """
        item can be a tuple or list of ints or strings, or a single
        int or string. String item is a nested field name, e.g., "a", "a:b",
        "a:b:c". Int item is the index of a field at the first level of the
        Struct.
        """
        if isinstance(item, list) or isinstance(item, tuple):
            keys = list(viewkeys(self.fields))
            return Struct(
                * [
                    (
                        keys[k]
                        if isinstance(k, int) else k, self[k]
                    ) for k in item
                ]
            )
        elif isinstance(item, int):
            return next(islice(viewvalues(self.fields), item, None))
        else:
            field = self._get_field_by_nested_name(item)
            if field is None:
                raise KeyError('field "%s" not found' % (item))
            return field

    def get(self, item, default_value):
        """
        similar to python's dictionary get method, return field of item if found
        (i.e. self.item is valid) or otherwise return default_value

        it's a syntax suger of python's builtin getattr method
        """
        return getattr(self, item, default_value)

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(item)
        try:
            return super(Struct, self).__getattribute__("fields")[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        # Disable setting attributes after initialization to prevent false
        # impression of being able to overwrite a field.
        # Allowing setting internal states mainly so that _parent can be set
        # post initialization.
        if getattr(self, '_frozen', None) and not key.startswith('_'):
            raise TypeError('Struct.__setattr__() is disabled after __init__()')
        super(Struct, self).__setattr__(key, value)

    def __add__(self, other):
        """
        Allows to merge fields of two schema.Struct using '+' operator.
        If two Struct have common field names, the merge is conducted
        recursively. Here are examples:

        Example 1
        s1 = Struct(('a', Scalar()))
        s2 = Struct(('b', Scalar()))
        s1 + s2 == Struct(
            ('a', Scalar()),
            ('b', Scalar()),
        )

        Example 2
        s1 = Struct(
            ('a', Scalar()),
            ('b', Struct(('c', Scalar()))),
        )
        s2 = Struct(('b', Struct(('d', Scalar()))))
        s1 + s2 == Struct(
            ('a', Scalar()),
            ('b', Struct(
                ('c', Scalar()),
                ('d', Scalar()),
            )),
        )
        """
        if not isinstance(other, Struct):
            return NotImplemented

        children = OrderedDict(self.get_children())
        for name, right_field in other.get_children():
            if name not in children:
                children[name] = right_field
                continue
            left_field = children[name]
            if not (isinstance(left_field, Struct) and isinstance(right_field, Struct)):
                raise TypeError(
                    "Type of left_field, " + str(type(left_field)) +
                    ", and type of right_field, " +
                    str(type(right_field)) +
                    ", must both the Struct to allow merging of the field, " + name)
            children[name] = left_field + right_field

        return Struct(*(viewitems(children)))

    def __sub__(self, other):
        """
        Allows to remove common fields of two schema.Struct from self by
        using '-' operator. If two Struct have common field names, the
        removal is conducted recursively. If a child struct has no fields
        inside, it will be removed from its parent. Here are examples:

        Example 1
        s1 = Struct(
            ('a', Scalar()),
            ('b', Scalar()),
        )
        s2 = Struct(('a', Scalar()))
        s1 - s2 == Struct(('b', Scalar()))

        Example 2
        s1 = Struct(
            ('b', Struct(
                ('c', Scalar()),
                ('d', Scalar()),
            ))
        )
        s2 = Struct(
            ('b', Struct(('c', Scalar()))),
        )
        s1 - s2 == Struct(
            ('b', Struct(
                ('d', Scalar()),
            )),
        )

        Example 3
        s1 = Struct(
            ('a', Scalar()),
            ('b', Struct(
                ('d', Scalar()),
            ))
        )
        s2 = Struct(
            ('b', Struct(
                ('c', Scalar())
                ('d', Scalar())
            )),
        )
        s1 - s2 == Struct(
            ('a', Scalar()),
        )
        """
        if not isinstance(other, Struct):
            return NotImplemented

        children = OrderedDict(self.get_children())
        for name, right_field in other.get_children():
            if name in children:
                left_field = children[name]
                if type(left_field) == type(right_field):
                    if isinstance(left_field, Struct):
                        child = left_field - right_field
                        if child.get_children():
                            children[name] = child
                            continue
                    children.pop(name)
                else:
                    raise TypeError(
                        "Type of left_field, " + str(type(left_field)) +
                        ", is not the same as that of right_field, " +
                        str(type(right_field)) +
                        ", yet they have the same field name, " + name)
        return Struct(*(children.items()))


class Scalar(Field):
    """Represents a typed scalar or tensor of fixed shape.

    A Scalar is a leaf in a schema tree, translating to exactly one tensor in
    the dataset's underlying storage.

    Usually, the tensor storing the actual values of this field is a 1D tensor,
    representing a series of values in its domain. It is possible however to
    have higher rank values stored as a Scalar, as long as all entries have
    the same shape.

    E.g.:

        Scalar(np.float64)

            Scalar field of type float64. Caffe2 will expect readers and
            datasets to expose it as a 1D tensor of doubles (vector), where
            the size of the vector is determined by this fields' domain.

        Scalar((np.int32, 5))

            Tensor field of type int32. Caffe2 will expect readers and
            datasets to implement it as a 2D tensor (matrix) of shape (L, 5),
            where L is determined by this fields' domain.

        Scalar((str, (10, 20)))

            Tensor field of type str. Caffe2 will expect readers and
            datasets to implement it as a 3D tensor of shape (L, 10, 20),
            where L is determined by this fields' domain.

    If the field type is unknown at construction time, call Scalar(), that will
    default to np.void as its dtype.

    It is an error to pass a structured dtype to Scalar, since it would contain
    more than one field. Instead, use from_dtype, which will construct
    a nested `Struct` field reflecting the given dtype's structure.

    A Scalar can also contain a blob, which represents the value of this
    Scalar. A blob can be either a numpy.ndarray, in which case it contain the
    actual contents of the Scalar, or a BlobReference, which represents a
    blob living in a caffe2 Workspace. If blob of different types are passed,
    a conversion to numpy.ndarray is attempted.
    """

    __slots__: Sequence[str] = ("_metadata", "dtype", "_original_dtype", "_blob")

    def __init__(self, dtype=None, blob=None, metadata=None):
        self._metadata = None
        self.set(dtype, blob, metadata, unsafe=True)
        super(Scalar, self).__init__([])

    def field_names(self):
        return ['']

    def field_type(self):
        return self.dtype

    def field_types(self):
        return [self.dtype]

    def field_metadata(self):
        return [self._metadata]

    def has_blobs(self):
        return self._blob is not None

    def field_blobs(self):
        assert self._blob is not None, 'Value is not set for this field.'
        return [self._blob]

    def all_scalars(self):
        return [self]

    def clone(self, keep_blobs=True):
        return Scalar(
            dtype=self._original_dtype,
            blob=self._blob if keep_blobs else None,
            metadata=self._metadata
        )

    def get(self):
        """Gets the current blob of this Scalar field."""
        assert self._blob is not None, 'Value is not set for this field.'
        return self._blob

    def __call__(self):
        """Shortcut for self.get()"""
        return self.get()

    @property
    def metadata(self):
        return self._metadata

    def set_metadata(self, value):
        assert isinstance(value, Metadata), \
            'metadata must be Metadata, got {}'.format(type(value))
        self._metadata = value
        self._validate_metadata()

    def _validate_metadata(self):
        if self._metadata is None:
            return
        if (self._metadata.categorical_limit is not None and
                self.dtype is not None):
            assert np.issubdtype(self.dtype, np.integer), \
                "`categorical_limit` can be specified only in integral " + \
                "fields but got {}".format(self.dtype)

    def set_value(self, blob, throw_on_type_mismatch=False, unsafe=False):
        """Sets only the blob field still validating the existing dtype"""
        if self.dtype.base != np.void and throw_on_type_mismatch:
            assert isinstance(blob, np.ndarray), "Got {!r}".format(blob)
            assert blob.dtype.base == self.dtype.base, (
                "Expected {}, got {}".format(self.dtype.base, blob.dtype.base))
        self.set(dtype=self._original_dtype, blob=blob, unsafe=unsafe)

    def set(self, dtype=None, blob=None, metadata=None, unsafe=False):
        """Set the type and/or blob of this scalar. See __init__ for details.

        Args:
            dtype: can be any numpy type. If not provided and `blob` is
                   provided, it will be inferred. If no argument is provided,
                   this Scalar will be of type np.void.
            blob:  if provided, can be either a BlobReference or a
                   numpy.ndarray. If a value of different type is passed,
                   a conversion to numpy.ndarray is attempted. Strings aren't
                   accepted, since they can be ambiguous. If you want to pass
                   a string, to either BlobReference(blob) or np.array(blob).
            metadata: optional instance of Metadata, if provided overrides
                      the metadata information of the scalar
        """
        if not unsafe:
            logger.warning(
                "Scalar should be considered immutable. Only call Scalar.set() "
                "on newly created Scalar with unsafe=True. This will become an "
                "error soon."
            )
        if blob is not None and isinstance(blob, basestring):
            raise ValueError(
                'Passing str blob to Scalar.set() is ambiguous. '
                'Do either set(blob=np.array(blob)) or '
                'set(blob=BlobReference(blob))'
            )

        self._original_dtype = dtype
        # Numpy will collapse a shape of 1 into an unindexed data array (shape = ()),
        # which betrays the docstring of this class (which expects shape = (1,)).
        # >>> import numpy as np
        # >> np.dtype((np.int32, 1))
        # dtype('int32')
        # >>> np.dtype((np.int32, 5))
        # dtype(('<i4', (5,)))
        if dtype is not None and isinstance(dtype, tuple) and dtype[1] == 1:
            dtype = (dtype[0], (1,))
        if dtype is not None:
            if isinstance(dtype, tuple) and dtype[0] == np.void:
                raise TypeError(
                    "Cannot set the Scalar with type {} for blob {}."
                    "If this blob is the output of some operation, "
                    "please verify the input of that operation has "
                    "proper type.".format(dtype, blob)
                )
            dtype = np.dtype(dtype)
        # If blob is not None and it is not a BlobReference, we assume that
        # it is actual tensor data, so we will try to cast it to a numpy array.
        if blob is not None and not isinstance(blob, BlobReference):
            preserve_shape = isinstance(blob, np.ndarray)
            if dtype is not None and dtype != np.void:
                blob = np.array(blob, dtype=dtype.base)
                # if array is empty we may need to reshape a little
                if blob.size == 0 and not preserve_shape:
                    blob = blob.reshape((0, ) + dtype.shape)
            else:
                assert isinstance(blob, np.ndarray), (
                    'Invalid blob type: %s' % str(type(blob)))

            # reshape scalars into 1D arrays
            # TODO(azzolini): figure out better way of representing this
            if len(blob.shape) == 0 and not preserve_shape:
                blob = blob.reshape((1, ))

            # infer inner shape from the blob given
            # TODO(dzhulgakov): tweak this to make it work with PackedStruct
            if (len(blob.shape) > 1 and dtype is not None and
                    dtype.base != np.void):
                dtype = np.dtype((dtype.base, blob.shape[1:]))
        # if we were still unable to infer the dtype
        if dtype is None:
            dtype = np.dtype(np.void)
        assert not dtype.fields, (
            'Cannot create Scalar with a structured dtype. ' +
            'Use from_dtype instead.'
        )
        self.dtype = dtype
        self._blob = blob
        if metadata is not None:
            self.set_metadata(metadata)
        self._validate_metadata()

    def set_type(self, dtype):
        self._original_dtype = dtype
        if dtype is not None:
            self.dtype = np.dtype(dtype)
        else:
            self.dtype = np.dtype(np.void)
        self._validate_metadata()

    def _pprint_impl(self, indent, str_buffer):
        str_buffer.write('  ' * (indent) +
            'Scalar({!r}, {!r}, {!r})'.format(
            self.dtype, self._blob, self._metadata) + "\n")

    def id(self):
        """
        Return the zero-indexed position of this scalar field in its schema.
        Used in order to index into the field_blob list returned by readers or
        accepted by writers.
        """
        return self._child_base_id()


def Map(
    keys,
    values,
    keys_name='keys',
    values_name='values',
    lengths_blob=None
):
    """A map is a List of Struct containing keys and values fields.
    Optionally, you can provide custom name for the key and value fields.
    """
    return List(
        Struct((keys_name, keys), (values_name, values)),
        lengths_blob=lengths_blob
    )

def MapWithEvicted(
    keys,
    values,
    keys_name='keys',
    values_name='values',
    lengths_blob=None,
    evicted_values=None
):
    """A map with extra field evicted_values
    """
    return ListWithEvicted(
        Struct((keys_name, keys), (values_name, values)),
        lengths_blob=lengths_blob,
        evicted_values=evicted_values
    )


def NamedTuple(name_prefix, *fields):
    return Struct(* [('%s_%d' % (name_prefix, i), field)
                     for i, field in enumerate(fields)])


def Tuple(*fields):
    """
    Creates a Struct with default, sequential, field names of given types.
    """
    return NamedTuple('field', *fields)


def RawTuple(num_fields, name_prefix='field'):
    """
    Creates a tuple of `num_field` untyped scalars.
    """
    assert isinstance(num_fields, int)
    assert num_fields >= 0
    return NamedTuple(name_prefix, *([np.void] * num_fields))


def from_dtype(dtype, _outer_shape=()):
    """Constructs a Caffe2 schema from the given numpy's dtype.

    Numpy supports scalar, array-like and structured datatypes, as long as
    all the shapes are fixed. This function breaks down the given dtype into
    a Caffe2 schema containing `Struct` and `Scalar` types.

    Fields containing byte offsets are not currently supported.
    """
    if not isinstance(dtype, np.dtype):
        # wrap into a ndtype
        shape = _outer_shape
        dtype = np.dtype((dtype, _outer_shape))
    else:
        # concatenate shapes if necessary
        shape = _outer_shape + dtype.shape
        if shape != dtype.shape:
            dtype = np.dtype((dtype.base, shape))

    if not dtype.fields:
        return Scalar(dtype)

    struct_fields = []
    for name, (fdtype, offset) in dtype.fields:
        assert offset == 0, ('Fields with byte offsets are not supported.')
        struct_fields += (name, from_dtype(fdtype, _outer_shape=shape))
    return Struct(*struct_fields)


class _SchemaNode(object):
    """This is a private class used to represent a Schema Node"""

    __slots__: Sequence[str] = ("name", "children", "type_str", "field")

    def __init__(self, name, type_str=''):
        self.name = name
        self.children = []
        self.type_str = type_str
        self.field = None

    def add_child(self, name, type_str=''):
        for child in self.children:
            if child.name == name and child.type_str == type_str:
                return child
        child = _SchemaNode(name, type_str)
        self.children.append(child)
        return child

    def get_field(self):

        list_names = ['lengths', 'values']
        map_names = ['lengths', 'keys', 'values']

        if len(self.children) == 0 or self.field is not None:
            if self.field is None:
                return Struct()
            else:
                return self.field

        child_names = []
        for child in self.children:
            child_names.append(child.name)

        if (set(child_names) == set(list_names)):
            for child in self.children:
                if child.name == 'values':
                    values_field = child.get_field()
                else:
                    lengths_field = child.get_field()
            self.field = List(
                values_field,
                lengths_blob=lengths_field
            )
            self.type_str = "List"
            return self.field
        elif (set(child_names) == set(map_names)):
            for child in self.children:
                if child.name == 'keys':
                    key_field = child.get_field()
                elif child.name == 'values':
                    values_field = child.get_field()
                else:
                    lengths_field = child.get_field()
            self.field = Map(
                key_field,
                values_field,
                lengths_blob=lengths_field
            )
            self.type_str = "Map"
            return self.field

        else:
            struct_fields = []
            for child in self.children:
                struct_fields.append((child.name, child.get_field()))

            self.field = Struct(*struct_fields)
            self.type_str = "Struct"
            return self.field

    def print_recursively(self):
        for child in self.children:
            child.print_recursively()
        logger.info("Printing node: Name and type")
        logger.info(self.name)
        logger.info(self.type_str)


def from_column_list(
    col_names, col_types=None,
    col_blobs=None, col_metadata=None
):
    """
    Given a list of names, types, and optionally values, construct a Schema.
    """
    if col_types is None:
        col_types = [None] * len(col_names)
    if col_metadata is None:
        col_metadata = [None] * len(col_names)
    if col_blobs is None:
        col_blobs = [None] * len(col_names)
    assert len(col_names) == len(col_types), (
        'col_names and col_types must have the same length.'
    )
    assert len(col_names) == len(col_metadata), (
        'col_names and col_metadata must have the same length.'
    )
    assert len(col_names) == len(col_blobs), (
        'col_names and col_blobs must have the same length.'
    )
    root = _SchemaNode('root', 'Struct')
    for col_name, col_type, col_blob, col_metadata in zip(
        col_names, col_types, col_blobs, col_metadata
    ):
        columns = col_name.split(FIELD_SEPARATOR)
        current = root
        for i in range(len(columns)):
            name = columns[i]
            type_str = ''
            field = None
            if i == len(columns) - 1:
                type_str = col_type
                field = Scalar(
                    dtype=col_type,
                    blob=col_blob,
                    metadata=col_metadata
                )
            next = current.add_child(name, type_str)
            if field is not None:
                next.field = field
            current = next

    return root.get_field()


def from_blob_list(schema, values, throw_on_type_mismatch=False):
    """
    Create a schema that clones the given schema, but containing the given
    list of values.
    """
    assert isinstance(schema, Field), 'Argument `schema` must be a Field.'
    if isinstance(values, BlobReference):
        values = [values]
    record = schema.clone_schema()
    scalars = record.all_scalars()
    assert len(scalars) == len(values), (
        'Values must have %d elements, got %d.' % (len(scalars), len(values))
    )
    for scalar, value in zip(scalars, values):
        scalar.set_value(value, throw_on_type_mismatch, unsafe=True)
    return record


def as_record(value):
    if isinstance(value, Field):
        return value
    elif isinstance(value, list) or isinstance(value, tuple):
        is_field_list = all(
            f is tuple and len(f) == 2 and isinstance(f[0], basestring)
            for f in value
        )
        if is_field_list:
            return Struct(* [(k, as_record(v)) for k, v in value])
        else:
            return Tuple(* [as_record(f) for f in value])
    elif isinstance(value, dict):
        return Struct(* [(k, as_record(v)) for k, v in viewitems(value)])
    else:
        return _normalize_field(value)


def FetchRecord(blob_record, ws=None, throw_on_type_mismatch=False):
    """
    Given a record containing BlobReferences, return a new record with same
    schema, containing numpy arrays, fetched from the current active workspace.
    """

    def fetch(v):
        if ws is None:
            return workspace.FetchBlob(str(v))
        else:
            return ws.blobs[str(v)].fetch()

    assert isinstance(blob_record, Field)
    field_blobs = blob_record.field_blobs()
    assert all(isinstance(v, BlobReference) for v in field_blobs)
    field_arrays = [fetch(value) for value in field_blobs]
    return from_blob_list(blob_record, field_arrays, throw_on_type_mismatch)


def FeedRecord(blob_record, arrays, ws=None):
    """
    Given a Record containing blob_references and arrays, which is either
    a list of numpy arrays or a Record containing numpy arrays, feeds the
    record to the current workspace.
    """

    def feed(b, v):
        if ws is None:
            workspace.FeedBlob(str(b), v)
        else:
            ws.create_blob(str(b))
            ws.blobs[str(b)].feed(v)
    assert isinstance(blob_record, Field)
    field_blobs = blob_record.field_blobs()
    assert all(isinstance(v, BlobReference) for v in field_blobs)
    if isinstance(arrays, Field):
        # TODO: check schema
        arrays = arrays.field_blobs()
    assert len(arrays) == len(field_blobs), (
        'Values must contain exactly %d ndarrays.' % len(field_blobs)
    )
    for blob, array in zip(field_blobs, arrays):
        feed(blob, array)


def NewRecord(net, schema):
    """
    Given a record of np.arrays, create a BlobReference for each one of them,
    returning a record containing BlobReferences. The name of each returned blob
    is NextScopedBlob(field_name), which guarantees unique name in the current
    net. Use NameScope explicitly to avoid name conflictions between different
    nets.
    """
    if isinstance(schema, Scalar):
        result = schema.clone()
        result.set_value(
            blob=net.NextScopedBlob('unnamed_scalar'),
            unsafe=True,
        )
        return result

    assert isinstance(schema, Field), 'Record must be a schema.Field instance.'
    blob_refs = [
        net.NextScopedBlob(prefix=name)
        for name in schema.field_names()
    ]
    return from_blob_list(schema, blob_refs)


def ConstRecord(net, array_record):
    """
    Given a record of arrays, returns a record of blobs,
    initialized with net.Const.
    """
    blob_record = NewRecord(net, array_record)
    for blob, array in zip(
        blob_record.field_blobs(), array_record.field_blobs()
    ):
        net.Const(array, blob)
    return blob_record


def InitEmptyRecord(net, schema_or_record, enforce_types=False):
    if not schema_or_record.has_blobs():
        record = NewRecord(net, schema_or_record)
    else:
        record = schema_or_record

    for blob_type, blob in zip(record.field_types(), record.field_blobs()):
        try:
            data_type = data_type_for_dtype(blob_type)
            shape = [0] + list(blob_type.shape)
            net.ConstantFill([], blob, shape=shape, dtype=data_type)
        except TypeError:
            logger.warning("Blob {} has type error".format(blob))
            # If data_type_for_dtype doesn't know how to resolve given numpy
            # type to core.DataType, that function can throw type error (for
            # example that would happen for cases of unknown types such as
            # np.void). This is not a problem for cases when the record if going
            # to be overwritten by some operator later, though it might be an
            # issue for type/shape inference.
            if enforce_types:
                raise
            # If we don't enforce types for all items we'll create a blob with
            # the default ConstantFill (FLOAT, no shape)
            net.ConstantFill([], blob, shape=[0])

    return record


_DATA_TYPE_FOR_DTYPE = [
    (np.str, core.DataType.STRING),
    (np.float16, core.DataType.FLOAT16),
    (np.float32, core.DataType.FLOAT),
    (np.float64, core.DataType.DOUBLE),
    (np.bool, core.DataType.BOOL),
    (np.int8, core.DataType.INT8),
    (np.int16, core.DataType.INT16),
    (np.int32, core.DataType.INT32),
    (np.int64, core.DataType.INT64),
    (np.uint8, core.DataType.UINT8),
    (np.uint16, core.DataType.UINT16),
]


def is_schema_subset(schema, original_schema):
    # TODO add more checks
    return set(schema.field_names()).issubset(
        set(original_schema.field_names()))

def equal_schemas(schema,
                  original_schema,
                  check_field_names=True,
                  check_field_types=True,
                  check_field_metas=False):
    assert isinstance(schema, Field)
    assert isinstance(original_schema, Field)

    if check_field_names and (
            schema.field_names() != original_schema.field_names()):
        return False
    if check_field_types and (
            schema.field_types() != original_schema.field_types()):
        return False
    if check_field_metas and (
            schema.field_metadata() != original_schema.field_metadata()):
        return False

    return True


def schema_check(schema, previous=None):
    record = as_record(schema)
    if previous is not None:
        assert equal_schemas(schema, previous)
    return record


def data_type_for_dtype(dtype):
    for np_type, dt in _DATA_TYPE_FOR_DTYPE:
        if dtype.base == np_type:
            return dt
    raise TypeError('Unknown dtype: ' + str(dtype.base))


def dtype_for_core_type(core_type):
    for np_type, dt in _DATA_TYPE_FOR_DTYPE:
        if dt == core_type:
            return np_type
    raise TypeError('Unknown core type: ' + str(core_type))


def attach_metadata_to_scalars(field, metadata):
    for f in field.all_scalars():
        f.set_metadata(metadata)
