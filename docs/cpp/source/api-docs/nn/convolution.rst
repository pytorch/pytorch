Convolution Layers
==================

Convolutional layers apply learnable filters to input data, extracting local features
through sliding window operations. They are fundamental to CNNs for image, audio, and
sequential data processing.

- **Conv1d/2d/3d**: Standard convolution for 1D sequences, 2D images, or 3D volumes
- **ConvTranspose1d/2d/3d**: Transposed convolution (deconvolution) for upsampling

**Key parameters:**

- ``in_channels``: Number of input channels
- ``out_channels``: Number of output channels (number of filters)
- ``kernel_size``: Size of the convolving kernel
- ``stride``: Stride of the convolution (default: 1)
- ``padding``: Zero-padding added to input (default: 0)
- ``dilation``: Spacing between kernel elements (default: 1)
- ``groups``: Number of blocked connections (default: 1, use ``in_channels`` for depthwise)

Conv1d
------

Applies 1D convolution over an input signal composed of several input planes.

.. doxygenclass:: torch::nn::Conv1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::Conv1dImpl
   :members:
   :undoc-members:

Conv2d
------

Applies 2D convolution over an input image. The most commonly used layer for
image processing tasks.

.. doxygenclass:: torch::nn::Conv2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::Conv2dImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   // Create Conv2d: 3 input channels, 64 output channels, 3x3 kernel
   auto conv = torch::nn::Conv2d(
       torch::nn::Conv2dOptions(3, 64, 3)
           .stride(1)
           .padding(1)
           .bias(true));

   auto output = conv->forward(input);  // input: [N, 3, H, W]

Conv3d
------

Applies 3D convolution over an input volume (e.g., video frames or 3D medical images).

.. doxygenclass:: torch::nn::Conv3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::Conv3dImpl
   :members:
   :undoc-members:

ConvTranspose1d
---------------

Applies 1D transposed convolution (fractionally-strided convolution) for upsampling.

.. doxygenclass:: torch::nn::ConvTranspose1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ConvTranspose1dImpl
   :members:
   :undoc-members:

ConvTranspose2d
---------------

Applies 2D transposed convolution for upsampling. Commonly used in decoder
networks and generative models.

.. doxygenclass:: torch::nn::ConvTranspose2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ConvTranspose2dImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   // Create ConvTranspose2d for upsampling
   auto conv_transpose = torch::nn::ConvTranspose2d(
       torch::nn::ConvTranspose2dOptions(64, 32, 4)
           .stride(2)
           .padding(1));

ConvTranspose3d
---------------

Applies 3D transposed convolution for upsampling volumetric data.

.. doxygenclass:: torch::nn::ConvTranspose3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ConvTranspose3dImpl
   :members:
   :undoc-members:
