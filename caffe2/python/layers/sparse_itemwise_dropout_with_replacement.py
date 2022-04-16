




from caffe2.python import schema
from caffe2.python.layers.layers import (
    IdList,
    ModelLayer,
)

# Model layer for implementing probabilistic replacement of individual elements in
# IdLists.  Takes probabilities for train, eval and predict nets as input, as
# well as the replacement value when dropout happens.  For features we may have
# available to us in train net but not in predict net, we'd set dropout
# probability for predict net to be 1.0 and set the feature to the replacement
# value given here.  This way, the value is tied to the particular model and not
# to any specific logic in feature processing in serving.

# Consider the following example where X is the values in the IdList and Lengths
# is the number of values corresponding to each example.
# X: [1, 2, 3, 4, 5]
# Lengths: [2, 3]
# This IdList contains 2 IdList features of lengths 2, 3.  Let's assume we used a
# ratio of 0.5 and ended up dropping out 2nd item in 2nd IdList feature, and used a
# replacement value of -1. We will end up with the following IdList.

# Y: [1, 2, 3, -1, 5]
# OutputLengths: [2, 3]
# where the 2nd item in 2nd IdList feature [4] was replaced with [-1].

class SparseItemwiseDropoutWithReplacement(ModelLayer):
    def __init__(
            self,
            model,
            input_record,
            dropout_prob_train,
            dropout_prob_eval,
            dropout_prob_predict,
            replacement_value,
            name='sparse_itemwise_dropout',
            **kwargs):

        super(SparseItemwiseDropoutWithReplacement, self).__init__(model, name, input_record, **kwargs)
        assert schema.equal_schemas(input_record, IdList), "Incorrect input type"

        self.dropout_prob_train = float(dropout_prob_train)
        self.dropout_prob_eval = float(dropout_prob_eval)
        self.dropout_prob_predict = float(dropout_prob_predict)
        self.replacement_value = int(replacement_value)
        assert (self.dropout_prob_train >= 0 and
                self.dropout_prob_train <= 1.0), \
            "Expected 0 <= dropout_prob_train <= 1, but got %s" \
            % self.dropout_prob_train
        assert (self.dropout_prob_eval >= 0 and
                self.dropout_prob_eval <= 1.0), \
            "Expected 0 <= dropout_prob_eval <= 1, but got %s" \
            % dropout_prob_eval
        assert (self.dropout_prob_predict >= 0 and
                self.dropout_prob_predict <= 1.0), \
            "Expected 0 <= dropout_prob_predict <= 1, but got %s" \
            % dropout_prob_predict
        assert(self.dropout_prob_train > 0 or
               self.dropout_prob_eval > 0 or
               self.dropout_prob_predict > 0), \
            "Ratios all set to 0.0 for train, eval and predict"

        self.output_schema = schema.NewRecord(model.net, IdList)
        if input_record.lengths.metadata:
            self.output_schema.lengths.set_metadata(
                input_record.lengths.metadata)
        if input_record.items.metadata:
            self.output_schema.items.set_metadata(
                input_record.items.metadata)

    def _add_ops(self, net, ratio):
        input_values_blob = self.input_record.items()
        input_lengths_blob = self.input_record.lengths()

        output_lengths_blob = self.output_schema.lengths()
        output_values_blob = self.output_schema.items()

        net.SparseItemwiseDropoutWithReplacement(
            [
                input_values_blob,
                input_lengths_blob
            ],
            [
                output_values_blob,
                output_lengths_blob
            ],
            ratio=ratio,
            replacement_value=self.replacement_value
        )

    def add_train_ops(self, net):
        self._add_ops(net, self.dropout_prob_train)

    def add_eval_ops(self, net):
        self._add_ops(net, self.dropout_prob_eval)

    def add_ops(self, net):
        self._add_ops(net, self.dropout_prob_predict)
