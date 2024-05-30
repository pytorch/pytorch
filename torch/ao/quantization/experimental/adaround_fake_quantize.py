from typing import Tuple

import torch
from torch.ao.quantization.fake_quantize import _is_symmetric_quant
from torch.ao.quantization.utils import is_per_tensor
from torch.quantization import FakeQuantize
from torch.quantization.observer import MinMaxObserver


class AdaroundFakeQuantizer(FakeQuantize):
    """
    This is a FakeQuantizer that enables an adaptive rounding fake quantizer.
    Adaround is a technique to adaptively round weights, derived from the paper https://arxiv.org/pdf/2004.10568.pdf
    For HTP compatibility, we are targeting to use symmetric quantization
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    V: torch.nn.Parameter

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        observer=MinMaxObserver,
        qscheme=torch.per_tensor_symmetric,  # not used, but needed for fakequant
        quant_min: int = -128,
        quant_max: int = 127,
        ch_axis: int = 0,
        # pyre-fixme[2]: Parameter must be annotated.
        **observer_kwargs,
    ):
        super().__init__(
            observer=observer,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_dynamic=False,
            **observer_kwargs,
        )
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert (
                quant_min <= quant_max
            ), "quant_min must be less than or equal to quant_max"
        # pyre-fixme[4]: Attribute must be annotated.
        self.qscheme = qscheme
        self.is_per_tensor: bool = is_per_tensor(qscheme)
        self.is_symmetric: bool = _is_symmetric_quant(qscheme)
        assert self.is_symmetric, "Only symmetric quantization is supported"
        self.ch_axis: int = ch_axis

        self.scale = torch.tensor([], requires_grad=False)
        self.zero_point = torch.tensor([], requires_grad=False)
        self.V = torch.nn.Parameter(torch.tensor([]), requires_grad=True)
        # Fixed Stretch parameters
        self.zeta: torch.Tensor = torch.tensor(1.1, requires_grad=False)
        self.gamma: torch.Tensor = torch.tensor(-0.1, requires_grad=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_soft_rounding = True

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self) -> str:
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}, (self.V >= 0).int().sum()={(self.V >= 0).int().sum()}"
        )

    def enable_weight_fake_quant(self) -> None:
        self.fake_quant_enabled[0] = 1

    def get_rectified_sigmoid_func(self) -> torch.Tensor:
        if self.use_soft_rounding:
            return torch.clamp(
                self.sigmoid(self.V) * (self.zeta - self.gamma) + self.gamma,
                min=0,
                max=1,
            )
        else:
            # This will dump a binary solution
            return (self.V >= 0).int()

    @torch.jit.ignore
    def update_scale(
        self, X: torch.Tensor, _scale: torch.Tensor, _zero_point: torch.Tensor
    ) -> None:
        if self.scale.numel() == 0:
            self.scale.data = _scale.to(X.device)
            self.zero_point = _zero_point.to(X.device)
        else:
            self.scale.data = _scale
            if not self.is_symmetric:
                self.zero_point = _zero_point
            else:
                self.zero_point = torch.zeros_like(_zero_point)
            for i in range(X.dim()):
                if i == self.ch_axis:
                    continue
                self.zero_point = self.zero_point.unsqueeze(i)
        X_q = X / self.scale
        X_q_floor = torch.floor(X_q)
        residual = X_q - X_q_floor  # [0,1)
        assert torch.all(
            torch.ge(residual, 0)
        ), "residual should be non-negative [0, 1)"
        V_init = -torch.log((self.zeta - self.gamma) / (residual - self.gamma) - 1)
        self.V.data = V_init

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled[0] == 1:
            X_detached = X.detach()
            self.activation_post_process(X_detached)
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            dims = list(range(X.dim()))
            if not self.is_per_tensor:
                dims.remove(self.ch_axis)
            if not self.is_per_tensor:
                for i in range(X.dim()):
                    if i == self.ch_axis:
                        continue
                    _scale = _scale.unsqueeze(i)
                    _zero_point = _zero_point.unsqueeze(i)
            self.update_scale(X_detached, _scale, _zero_point)

        if self.fake_quant_enabled[0] == 1:
            # Perform soft quantization
            # See the equation (23) in Adaround paper
            h_v = self.get_rectified_sigmoid_func()
            X_q = X / self.scale
            # Straight-Through Estimator for floor function
            X_q_floor = torch.floor(X_q) + self.zero_point
            # Regardless of rounding, gradient should be able to flow back to self.V from X_q_dq.
            # With adaround, we don't train weight, but train V only.
            X_q_dq = (
                torch.clamp(X_q_floor + h_v, min=self.quant_min, max=self.quant_max)
                - self.zero_point
            ) * self.scale
            return X_q_dq
        else:
            return X
