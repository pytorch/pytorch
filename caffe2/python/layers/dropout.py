# Module caffe2.python.layers.dropout





from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer


class Dropout(ModelLayer):

    def __init__(
            self,
            model,
            input_record,
            name='dropout',
            ratio=0.5,
            dropout_for_eval=False,
            **kwargs):

        super(Dropout, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        assert (ratio >= 0 and ratio < 1.0), \
            "Expected 0 <= ratio < 1, but got ratio of %s" % ratio

        self.output_schema = input_record.clone_schema()
        self.output_schema.set_value(self.get_next_blob_reference('output'))
        self.dropout_for_eval = dropout_for_eval

        self.ratio = ratio

    def _add_ops(self, net, is_test):
        input_blob = self.input_record.field_blobs()
        output_blobs = self.output_schema.field_blobs() \
                     + [net.NextScopedBlob('d_mask')]

        net.Dropout(input_blob,
                    output_blobs,
                    ratio=self.ratio,
                    is_test=is_test)

    def add_train_ops(self, net):
        self._add_ops(net, is_test=False)

    def add_eval_ops(self, net):
        self._add_ops(net, is_test=(not self.dropout_for_eval))

    def add_ops(self, net):
        self.add_eval_ops(net)
