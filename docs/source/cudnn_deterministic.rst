.. note::

    In some circumstances when using the CUDA backend with CuDNN, this operator
    may select a nondeterministic algorithm to increase performance. If this is
    undesirable, you can try to make the operation deterministic (potentially at
    a performance cost) by setting ``torch.backends.cudnn.deterministic =
    True``.
    Please see the notes on :doc:`/notes/randomness` for background.
