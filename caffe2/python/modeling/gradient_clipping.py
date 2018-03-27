from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.proto import caffe2_pb2
from caffe2.python.optimizer import get_param_device
from caffe2.python.modeling.net_modifier import NetModifier

import logging

logger = logging.getLogger(__name__)


class GradientClipping(NetModifier):

    L1_NORM = 'l1_norm'
    L2_NORM = 'l2_norm'

    BY_NORM = 'by_norm'

    GRAD_CLIP_METHODS = [BY_NORM]
    CLIP_GRADIENT_NORM_TYPES = [L2_NORM, L1_NORM]

    def __init__(self, grad_clip_method, clip_norm_type, clip_threshold,
                 use_parameter_norm=False, compute_norm_ratio=False):
        """
        Clips gradient to avoid gradient magnitude explosion or vanishing gradient.

        Args:
        grad_clip_method: ways to clip the gradients
        clip_norm_type: type of norm used in the necessary computation
        clip_threshold: threshold used to determine whether to clip
        use_parameter_norm: a boolean to indicate whether to incorporate
            the norm of the parameter
        compute_norm_ratio: a boolean to compute the ratio between gradient norm
            and parameter norm explicitly for debugging purpose
        """

        assert grad_clip_method in self.GRAD_CLIP_METHODS, (
            "This method of clipping, {}, has not been implemented.".format(
                clip_norm_type))

        assert clip_norm_type in self.CLIP_GRADIENT_NORM_TYPES, (
            "This method of clipping, {}, has not been implemented.".format(
                clip_norm_type))

        self.grad_clip_method = grad_clip_method
        self.clip_norm_type = clip_norm_type
        self.clip_threshold = float(clip_threshold)
        self.use_parameter_norm = use_parameter_norm
        self.compute_norm_ratio = compute_norm_ratio

    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None):

        assert grad_map is not None

        CPU = core.DeviceOption(caffe2_pb2.CPU)

        for param, grad in grad_map.items():

            # currently sparse gradients won't be clipped
            # futher implementation is needed to enable it
            if isinstance(grad, core.GradientSlice):
                continue

            device = get_param_device(
                param,
                grad_map[str(param)],
                param_to_device=blob_to_device,
                default_device=CPU,
            )

            with core.DeviceScope(device):
                if self.grad_clip_method == self.BY_NORM:
                    if self.clip_norm_type == self.L2_NORM:
                        p = 2
                    elif self.clip_norm_type == self.L1_NORM:
                        p = 1

                    grad_norm = net.LpNorm(
                        [grad],
                        net.NextScopedBlob(prefix=str(grad) + '_l{}_norm'.format(p)),
                        p=p,
                    )

                    if p == 2:
                        grad_norm = net.Pow([grad_norm], exponent=0.5)

                    op_inputs = [grad, grad_norm]

                    if self.use_parameter_norm:
                        param_norm = net.LpNorm(
                            [param],
                            net.NextScopedBlob(
                                prefix=str(param) + '_l{}_norm'.format(p)),
                            p=p,
                        )

                        if p == 2:
                            param_norm = net.Pow([param_norm], exponent=0.5)

                        op_inputs.append(param_norm)

                        if self.compute_norm_ratio:
                            net.Div(
                                [grad_norm, param_norm],
                                [net.NextScopedBlob(
                                    prefix=str(param) + '_norm_ratio')]
                            )

                    net.ClipTensorByScaling(
                        op_inputs,
                        [grad],
                        threshold=self.clip_threshold,
                    )
