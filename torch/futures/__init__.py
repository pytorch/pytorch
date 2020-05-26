import torch


class Future(torch._C.Future):
    r"""Wrapper around a ``torch._C.Future``.
    """
    def __new__(cls):
        return super(Future, cls).__new__(cls)

    def wait(self):
        r"""
        Block until the value of this ``Future`` is ready.

        Return:
            The value held by this ``Future``.
        """
        return super(Future, self).wait()

    def then(self, callback):
        r"""
        Append the given callback function to this ``Future``, which will be run
        in order when the ``Future`` is completed.

        Argument:
            callback(``Callable``): a ``Callable`` that takes this ``Future`` as
                                    the only argument.

        Return:
            A new ``Future`` object that holds the return value of the
            ``callback`` and will be marked as completed when the given
            ``callback`` finishes.
        """
        return super(Future, self).then(callback)

    def set_result(self, result):
        r"""
        Mark this future as completed using the provided result.
        """
        super(Future, self).set_result(result)