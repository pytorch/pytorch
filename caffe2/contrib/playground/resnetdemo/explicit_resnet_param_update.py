from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2


def gen_param_update_builder_fun(self, model, dataset, is_train):
    if not is_train:
        return None
    else:
        # from sherlok
        for idx in range(self.opts['distributed']['first_xpu_id'],
                         self.opts['distributed']['first_xpu_id'] +
                         self.opts['distributed']['num_xpus']):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, idx)):
                workspace.CreateBlob('{}_{}/lr'.
                    format(self.opts['distributed']['device'], idx))

        def add_parameter_update_ops(model):
            model.Iter("ITER")
            weight_decay = model.param_init_net.ConstantFill(
                [], 'weight_decay', shape=[1],
                value=self.opts['model_param']['weight_decay']
            )
            weight_decay_bn = model.param_init_net.ConstantFill(
                [], 'weight_decay_bn', shape=[1],
                value=self.opts['model_param']['weight_decay_bn']
            )
            one = model.param_init_net.ConstantFill(
                [], "ONE", shape=[1], value=1.0
            )

            '''
            Add the momentum-SGD update.
            '''
            params = model.GetParams()
            assert(len(params) > 0)

            for param in params:
                param_grad = model.param_to_grad[param]
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0
                )

                if '_bn' in str(param):
                    model.WeightedSum(
                        [param_grad, one, param, weight_decay_bn], param_grad
                    )
                else:
                    model.WeightedSum(
                        [param_grad, one, param, weight_decay], param_grad
                    )

                # Update param_grad and param_momentum in place
                model.net.MomentumSGDUpdate(
                    [param_grad, param_momentum, 'lr', param],
                    [param_grad, param_momentum, param],
                    momentum=0.9,
                    nesterov=1
                )

        return add_parameter_update_ops
