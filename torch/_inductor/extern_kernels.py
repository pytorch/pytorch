from typing import Any


class KernelNamespace:
    def __getattr__(self, name: str) -> Any:
        # Extern kernels register themselves here as a side effect of importing
        # select_algorithm (and the lowerings it imports). Generated code loaded
        # from the cache can run before anything has imported it.
        import torch._inductor.select_algorithm  # noqa: F401

        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None


# these objects are imported from the generated wrapper code
extern_kernels = KernelNamespace()
