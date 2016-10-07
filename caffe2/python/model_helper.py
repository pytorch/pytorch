from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core

import logging


class ModelHelperBase(object):
    """A helper model so we can write models more easily, without having to
    manually define parameter initializations and operators separately.
    In order to add support for specific operators, inherit from this class
    and add corresponding methods. Operator representing methods should
    take care of adding their parameters to params
    """

    def __init__(self, name=None, init_params=True, allow_not_known_ops=True):
        if name is None:
            name = "model"
        self.net = core.Net(name)
        self.param_init_net = core.Net(name + '_init')

        self.param_to_grad = {}
        self.params = []
        self.gradient_ops_added = False
        self.init_params = init_params
        self.allow_not_known_ops = allow_not_known_ops

    def Proto(self):
        return self.net.Proto()

    def InitProto(self):
        return self.param_init_net.Proto()

    def RunAllOnGPU(self, *args, **kwargs):
        self.param_init_net.RunAllOnGPU(*args, **kwargs)
        self.net.RunAllOnGPU(*args, **kwargs)

    def CreateDB(self, blob_out, db, db_type, **kwargs):
        dbreader = self.param_init_net.CreateDB(
            [], blob_out, db=db, db_type=db_type, **kwargs)
        return dbreader

    def AddGradientOperators(self, *args, **kwargs):
        if self.gradient_ops_added:
            raise RuntimeError("You cannot run AddGradientOperators twice.")
        self.gradient_ops_added = True
        grad_map = self.net.AddGradientOperators(*args, **kwargs)
        for p in self.params:
            if str(p) in grad_map:
                self.param_to_grad[p] = grad_map[str(p)]
        return grad_map

    def TensorProtosDBInput(
        self, unused_blob_in, blob_out, batch_size, db, db_type, **kwargs
    ):
        """TensorProtosDBInput."""
        dbreader_name = "dbreader_" + db
        dbreader = self.param_init_net.CreateDB(
            [], dbreader_name,
            db=db, db_type=db_type)
        return self.net.TensorProtosDBInput(
            dbreader, blob_out, batch_size=batch_size)

    def AddOperator(self, op_type, inputs, parameters, *args, **kwargs):
        """
        Adds an operator to a model. Use parameters list
        to specify which operator inputs are model parameters to be
        optimized.

        Example of usage:

        model.SparseLengthsSum(
             [embedding, indices, lengths],
             parameters=[embedding],
        )

        Here embedding is a parameter to be optimized while indices
        and lengths are not.
        """

        extra_parameters = filter(lambda x: (x not in inputs), parameters)
        if len(extra_parameters) > 0:
            raise Exception("Some parameters are not inputs: {}".format(
                map(str, extra_parameters)
            ))

        self.params.extend(parameters)
        return self.net.__getattr__(op_type)(inputs, *args, **kwargs)

    def __getattr__(self, op_type):
        """Catch-all for all other operators, mostly those without params."""
        if not core.IsOperator(op_type):
            raise RuntimeError(
                'Method ' + op_type + ' is not a registered operator.'
            )
        # known_working_ops are operators that do not need special care.
        known_working_ops = [
            "Accuracy",
            "Adam",
            "AveragedLoss",
            "Cast",
            "EnsureCPUOutput",
            "LabelCrossEntropy",
            "LearningRate",
            "Print",
            "Sigmoid",
            "Scale",
            "Snapshot",
            "Softmax",
            "StopGradient",
            "Summarize",
            "Tanh",
            "WeightedSum",
            "SquaredL2Distance",
            "FlattenToVec",
            "NHWC2NCHW",
            "ScatterWeightedSum",
            "Squeeze",
            "NCCLAllreduce",
            "ConstantFill",
            "Add",
            "DequeueBlobs",
        ]
        if op_type not in known_working_ops:
            assert self.allow_not_known_ops
            logging.warning("You are creating an op that the ModelHelperBase "
                            "does not recognize: {}.".format(op_type))
        return self.net.__getattr__(op_type)
