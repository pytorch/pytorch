"""
Defines a minimal set of data types that allow to represent datasets with
arbitrary nested structure, including objects of variable length, such as
maps and lists.

This defines a columnar storage format for such datasets on top of caffe2
tensors. In terms of capacity of representation, it can represent most of
the data types supported by Parquet, ORC, DWRF file formats.

See comments in operator_test/dataset_ops_test.py for a example and
walkthrough on how to use schema to store and iterate through a structured
in-memory dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _join_field_name(prefix, suffix):
    if prefix and suffix:
        return '{}:{}'.format(prefix, suffix)
    elif prefix:
        return prefix
    elif suffix:
        return suffix
    else:
        return ''


class Field(object):
    """Represents an abstract field type in a dataset.
    """
    def __init__(self, children):
        """Derived classes must call this after their initialization."""
        self._parent = (None, 0)
        offset = 0
        self._field_offsets = []
        for child in children:
            self._field_offsets.append(offset)
            offset += len(child.field_names())
        self._field_offsets.append(offset)

    def field_names(self):
        """Return the children field names for this field."""
        raise NotImplementedError('Field is an abstract class.')

    def field_types(self):
        """Return the numpy.dtype for each of the children fields."""
        raise NotImplementedError('Field is an abstract class.')

    def clone(self):
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
        return ((self.field_names() == other.field_names()) and
                (self.field_types() == other.field_types()))

class List(Field):
    """Represents a variable-length list.

    Values of a list can also be complex fields such as Lists and Structs.
    In addition to the fields exposed by its `values` field, a List exposes an
    additional `lengths` field, which will contain the size of each list under
    the parent domain.
    """
    def __init__(self, values):
        assert isinstance(values, Field)
        self.lengths = Scalar(np.int32)
        self.values = values.clone()
        self.lengths._set_parent(self, 0)
        self.values._set_parent(self, 1)
        Field.__init__(self, [self.lengths, self.values])

    def field_names(self):
        value_fields = self.values.field_names()
        return (
            ['lengths'] +
            [_join_field_name('values', v) for v in value_fields])

    def field_types(self):
        return self.lengths.field_types() + self.values.field_types()

    def clone(self):
        return List(self.values)


class Struct(Field):
    """Represents a named list of fields sharing the same domain.
    """
    def __init__(self, *fields):
        for field in fields:
            assert len(field) == 2
            assert field[0], 'Field names cannot be empty'
            assert field[0] != 'lengths', (
                'Struct cannot contain a field named `lengths`.')
            assert isinstance(field[1], Field)
        fields = [(name, field.clone()) for name, field in fields]
        for id, (name, field) in enumerate(fields):
            field._set_parent(self, id)
        self.fields = OrderedDict(fields)
        Field.__init__(self, self.fields.values())

    def field_names(self):
        names = []
        for name, field in self.fields.items():
            names += [_join_field_name(name, f) for f in field.field_names()]
        return names

    def field_types(self):
        types = []
        for name, field in self.fields.items():
            types += field.field_types()
        return types

    def clone(self):
        return Struct(*self.fields.items())

    def __getattr__(self, item):
        return self.fields[item]


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

            Scalar field of type float32. Caffe2 will expect readers and
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
    """
    def __init__(self, dtype=None):
        self._original_dtype = dtype
        self.dtype = np.dtype(dtype or np.void)
        assert not self.dtype.fields, (
            'Cannot create Scalar with a structured dtype. ' +
            'Use from_dtype instead.')
        Field.__init__(self, [])

    def field_names(self):
        return ['']

    def field_types(self):
        return [self.dtype]

    def clone(self):
        return Scalar(self._original_dtype)

    def id(self):
        """
        Return the zero-indexed position of this scalar field in its schema.
        Used in order to index into the field_blob list returned by readers or
        accepted by writers.
        """
        return self._child_base_id()


def Map(keys, values, keys_name='keys', values_name='values'):
    """A map is a List of Struct containing keys and values fields.
    Optionally, you can provide custom name for the key and value fields.
    """
    return List(Struct((keys_name, keys), (values_name, values)))


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
            assert self.field is not None
            return self.field

        child_names = []
        for child in self.children:
            child_names.append(child.name)

        if (set(child_names) == set(list_names)):
            for child in self.children:
                if child.name == 'values':
                    self.field = List(child.get_field())
                    self.type_str = "List"
                    return self.field

        elif (set(child_names) == set(map_names)):
            for child in self.children:
                if child.name == 'keys':
                    key_field = child.get_field()
                elif child.name == 'values':
                    values_field = child.get_field()
            self.field = Map(key_field, values_field)
            self.type_str = "Map"
            return self.field

        else:
            struct_fields = []
            for child in self.children:
                if child.field is not None:
                    struct_fields.append((child.name, child.field))
                else:
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


def from_column_list(column_names, column_types):

    root = _SchemaNode('root', 'Struct')
    for column_name, column_type in zip(column_names, column_types):
        columns = column_name.split(':')
        current = root
        for i in range(len(columns)):
            name = columns[i]
            type_str = ''
            field = None
            if i == len(columns) - 1:
                type_str = column_type
                field = Scalar(column_type)
            next = current.add_child(name, type_str)
            if field is not None:
                next.field = field
            current = next

    return root.get_field()
