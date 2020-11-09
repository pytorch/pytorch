## @package gather_record
# Module caffe2.python.layers.gather_record





from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer


class GatherRecord(ModelLayer):
    """
    Given 1-D `indices` tensor, gather elements at `i` in `indices` from all the
    blobs in `record`. If a blob is a values blob of a list, all the elements
    included by the list's lengths blob are gathered. For example,

    Input:
        indices = [0, 2]
        record:a = [[0, 1], [2, 3], [4, 5], [6, 7]]
        record:b:lengths = [0, 1, 2, 3]
        record:b:items = [0, 1, 2, 3, 4, 5]

    Output:
        a = [[0, 1], [4, 5]]
        b:lengths = [0, 2]
        b:items = [1, 2]

    This supports nested list.
    """

    def __init__(self, model, input_record, name='gather_record', **kwargs):
        super(GatherRecord, self).__init__(model, name, input_record, **kwargs)

        assert 'indices' in input_record
        assert 'record' in input_record

        self.output_schema = schema.NewRecord(
            model.net, input_record.record.clone_schema())

        self._indices = self.input_record.indices()

    def _gather_scalar(self, net, record, lengths_blob, output_record):
        if lengths_blob is None:
            net.Gather([record(), self._indices], output_record())
        else:
            net.LengthsGather([record(), lengths_blob, self._indices],
                              output_record())

    def _gather_struct(self, net, record, lengths_blob, output_record):
        for name, field in record.get_children():
            self._dispatch(net, field, lengths_blob, output_record[name])

    def _gather_list(self, net, record, lengths_blob, output_record):
        self._gather_scalar(
            net, record.lengths, lengths_blob, output_record.lengths)
        if lengths_blob is None:
            lengths_blob = record.lengths()
        else:
            # TODO(kittipat): This is a hacky solution until LengthsSum for int
            # is implemented
            lengths_float = net.Cast(
                record.lengths(),
                net.NextScopedBlob(str(record.lengths()) + '_float'),
                to=core.DataType.FLOAT,
            )
            lengths_blob_float = net.LengthsSum(
                [lengths_float, lengths_blob],
                net.NextScopedBlob(str(record.lengths()) + "_nested_float")
            )
            lengths_blob = net.Cast(
                lengths_blob_float,
                net.NextScopedBlob(str(record.lengths()) + "_nested"),
                to=core.DataType.INT32,
            )
        self._dispatch(net, record._items, lengths_blob, output_record._items)

    def _dispatch(self, net, record, lengths_blob, output_record):
        if isinstance(record, schema.Scalar):
            self._gather_scalar(net, record, lengths_blob, output_record)
        elif isinstance(record, schema.Struct):
            self._gather_struct(net, record, lengths_blob, output_record)
        elif isinstance(record, schema.List):
            self._gather_list(net, record, lengths_blob, output_record)
        else:
            raise NotImplementedError

    def add_ops(self, net):
        self._dispatch(net, self.input_record.record, None, self.output_schema)
