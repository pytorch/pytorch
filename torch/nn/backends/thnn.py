class THNNFunctionBackend(object):
    def __getattr__(self, name):
        raise RuntimeError(
            "THNNFunctionBackend has been deprecated. You are seeing this error "
            "likely because you applied a patch dumped from a module saved by an "
            "older version of PyTorch in which THNNFunctionBackend hasn't been "
            "deprecated yet. To address this issue, you either have to revert "
            "the patch, or switch to an older version of PyTorch."
        )


def _get_thnn_function_backend():
    # type() -> None
    """
    Prior to the removal of THNN backend, nn.Module had a `_backend` member
    which was an instance of class THNNFunctionBackend whose __reduce__() was
    defined as follows:

        class THNNFunctionBackend(FunctionBackend):

            def __reduce__(self):
                return (_get_thnn_function_backend, ())

    This means that modules that were pickled prior to the removal of THNN
    backend require invoking _get_thnn_function_backend() for deserialization.
    Keeping the function to maintain backward compatibility.
    """
    return THNNFunctionBackend()
