import torch


class WrapperModule:
    """Wraps the instance of wrapped_type.
    For graph_mode traces the instance of wrapped_type.
    Randomaly initializes num_params tensors with single float element.
    Args:
        wrapped_type:
            - Object type to be wrapped.
                Expects the wrapped_type to:
                   - be constructed with pt_fn specified in module_config.
                   - provide forward method that takes module_config.num_params args.
        module_config:
            - Specified pt_fn to construct wrapped_type with, whether graph_mode
              is enabled, and number of parameters wrapped_type's forward method
              takes.
        debug:
            - Whether debug mode is enabled.
        save:
            - In graph mode, whether graph is to be saved.
    """

    def __init__(self, wrapped_type, module_config, debug, save=False):
        pt_fn = module_config.pt_fn
        self.module = wrapped_type(pt_fn)
        self.tensor_inputs = []
        self.module_name = wrapped_type.__name__
        for _ in range(module_config.num_params):
            self.tensor_inputs.append(torch.randn(1))
        if module_config.graph_mode:
            self.module = torch.jit.trace(self.module, self.tensor_inputs)
            if save:
                file_name = self.module_name + "_" + pt_fn.__name__ + ".pt"
                torch.jit.save(self.module, file_name)
                print(f"Generated graph is saved in {file_name}")
        print(
            f"Benchmarking module {self.module_name} with fn {pt_fn.__name__}: Graph mode:{module_config.graph_mode}"
        )
        if debug and isinstance(self.module, torch.jit.ScriptModule):
            print(self.module.graph)
            print(self.module.code)

    def forward(self, niters):
        with torch.no_grad():
            for _ in range(niters):
                self.module.forward(*self.tensor_inputs)
