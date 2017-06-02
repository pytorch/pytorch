torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms are common image transforms. They can be chained together using :class:`Compose`

.. autoclass:: Compose

Transforms on PIL.Image
-----------------------

.. autoclass:: Scale

.. autoclass:: CenterCrop

.. autoclass:: RandomCrop

.. autoclass:: RandomHorizontalFlip

.. autoclass:: RandomSizedCrop

.. autoclass:: Pad

Transforms on torch.\*Tensor
----------------------------

.. autoclass:: Normalize
	:members: __call__
	:special-members:


Conversion Transforms
---------------------

.. autoclass:: ToTensor
	:members: __call__
	:special-members:

.. autoclass:: ToPILImage
	:members: __call__
	:special-members:

Generic Transforms
------------------

.. autoclass:: Lambda

