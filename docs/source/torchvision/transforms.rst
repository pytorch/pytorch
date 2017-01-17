torchvision.transforms
======================

| Transforms are common image transforms.
| They can be chained together using ``transforms.Compose``

``transforms.Compose``
~~~~~~~~~~~~~~~~~~~~~~

| One can compose several transforms together.
| For example.

.. code:: python

    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std = [ 0.229, 0.224, 0.225 ]),
    ])

Transforms on PIL.Image
-----------------------

``Scale(size, interpolation=Image.BILINEAR)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Rescales the input PIL.Image to the given ‘size’.
| ‘size’ will be the size of the smaller edge.

| For example, if height > width, then image will be
| rescaled to (size \* height / width, size)

-  size: size of the smaller edge
-  interpolation: Default: PIL.Image.BILINEAR

``CenterCrop(size)`` - center-crops the image to the given size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Crops the given PIL.Image at the center to have a region of
| the given size. size can be a tuple (target\_height, target\_width)
| or an integer, in which case the target will be of a square shape
  (size, size)

``RandomCrop(size, padding=0)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Crops the given PIL.Image at a random location to have a region of
| the given size. size can be a tuple (target\_height, target\_width)
| or an integer, in which case the target will be of a square shape
  (size, size)
| If ``padding`` is non-zero, then the image is first zero-padded on
  each side with ``padding`` pixels.

``RandomHorizontalFlip()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Randomly horizontally flips the given PIL.Image with a probability of
0.5

``RandomSizedCrop(size, interpolation=Image.BILINEAR)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Random crop the given PIL.Image to a random size of (0.08 to 1.0) of
  the original size
| and and a random aspect ratio of 3/4 to 4/3 of the original aspect
  ratio

This is popularly used to train the Inception networks

-  size: size of the smaller edge
-  interpolation: Default: PIL.Image.BILINEAR

``Pad(padding, fill=0)``
~~~~~~~~~~~~~~~~~~~~~~~~

| Pads the given image on each side with ``padding`` number of pixels,
  and the padding pixels are filled with
| pixel value ``fill``.
| If a ``5x5`` image is padded with ``padding=1`` then it becomes
  ``7x7``

Transforms on torch.\*Tensor
----------------------------

``Normalize(mean, std)``
~~~~~~~~~~~~~~~~~~~~~~~~

Given mean: (R, G, B) and std: (R, G, B), will normalize each channel of
the torch.\*Tensor, i.e. channel = (channel - mean) / std

Conversion Transforms
---------------------

-  ``ToTensor()`` - Converts a PIL.Image (RGB) or numpy.ndarray (H x W x
   C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W)
   in the range [0.0, 1.0]
-  ``ToPILImage()`` - Converts a torch.\*Tensor of range [0, 1] and
   shape C x H x W or numpy ndarray of dtype=uint8, range[0, 255] and
   shape H x W x C to a PIL.Image of range [0, 255]

Generic Transofrms
------------------

``Lambda(lambda)``
~~~~~~~~~~~~~~~~~~

| Given a Python lambda, applies it to the input ``img`` and returns it.
| For example:

.. code:: python

    transforms.Lambda(lambda x: x.add(10))
