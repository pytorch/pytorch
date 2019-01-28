from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def gen_param_update_builder_fun(self, model, dataset, is_train):
    if not is_train:
        return None
    else:
        def add_parameter_update_ops(model):
            model.AddWeightDecay(1e-4)
            ITER = model.Iter("ITER")
            stepsz = int(30 *
                         self.opts['epoch_iter']['num_train_sample_per_epoch'] /
                         self.total_batch_size)
            LR = model.net.LearningRate(
                [ITER],
                "lr",
                base_lr=self.opts['model_param']['base_learning_rate'],
                policy="step",
                stepsize=stepsz,
                gamma=0.1,
            )

            params = model.GetParams()
            assert(len(params) > 0)
            for param in params:
                param_grad = model.param_to_grad[param]
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0
                )

                # Update param_grad and param_momentum in place
                model.net.MomentumSGDUpdate(
                    [param_grad, param_momentum, LR, param],
                    [param_grad, param_momentum, param],
                    momentum=0.9,
                    nesterov=1
                )

        return add_parameter_update_ops
