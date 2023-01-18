Install functorch
=================

As of PyTorch 1.13, functorch is now included in the PyTorch binary and no
longer requires the installation of a separate functorch package. That is,
after installing PyTorch (`instructions <https://pytorch.org>`_),
you'll be able to ``import functorch`` in your program.

If you're upgrading from an older version of functorch (functorch 0.1.x or 0.2.x),
then you may need to uninstall functorch first via ``pip uninstall functorch``.

We've maintained backwards compatibility for ``pip install functorch``: this
command works for PyTorch 1.13 and will continue to work for the foreseeable future
until we do a proper deprecation. This is helpful if you're maintaining a library
that supports multiple versions of PyTorch and/or functorch.

Colab
-----

Please see `this colab for instructions. <https://colab.research.google.com/drive/1GNfb01W_xf8JRu78ZKoNnLqiwcrJrbYG#scrollTo=HJ1srOGeNCGA>`_

Nightly
-------

Looking for the newest functorch features? Please download the latest nightly PyTorch
binary (``import functorch`` is included in nightly PyTorch binaries as of 09/21/2022).
by following instructions `here <https://pytorch.org>`_.

Previous Versions
-----------------

For PyTorch 1.11.x and PyTorch 1.12.x:
Please first install PyTorch and then run the following command:

::

  pip install functorch
