torch.utils.tensorboard
===================================

.. warning::

    This code is EXPERIMENTAL and might change in the future. It also
    currently does not support all model types for ``add_graph``, which
    we are actively working on.

Before going further, more details on TensorBoard can be found at
https://www.tensorflow.org/tensorboard/

Once you've installed TensorBoard, these utilities let you log PyTorch models
and metrics into a directory for visualization within the TensorBoard UI.
Scalars, images, histograms, graphs, and embedding visualizations are all
supported for PyTorch models and tensors as well as Caffe2 nets and blobs.

The SummaryWriter class is your main entry to log data for consumption
and visualization by TensorBoard. For example:

.. code:: python


    import torch
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()

This can then be visualized with TensorBoard, which should be installable
and runnable with::

    pip install tb-nightly  # Until 1.14 moves to the release channel
    tensorboard --logdir=runs


The following shows how to log data in tensorboard from pytorch.

.. code:: python

    import torch
    import torchvision.utils as vutils
    import numpy as np  # you can use torch tensor instead of numpy array.
    from torch.utils.tensorboard import SummaryWriter


    writer = SummaryWriter()
    sample_rate = 44100
    freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

    for n_iter in range(100):  # n_iter is used as the x axis of the plot.

        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        # data can be grouped by `slash`
        writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
        writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

        # plot many lines in one graph
        writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                                'xcosx': n_iter * np.cos(n_iter),
                                                'arctanx': np.arctan(n_iter)}, n_iter)

        dummy_img = torch.rand(32, 3, 64, 64)  # your input to the network
        if n_iter % 10 == 0:  # saving images cost you computing power and disk space, only save when needed.
            x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
            writer.add_image('Image_vutils_mosaic', x, n_iter)
            writer.add_images('Image_implicit_mosaic', dummy_img, n_iter)  # note the 's'

            dummy_audio = torch.zeros(sample_rate * 2)
            for i in range(x.size(0)):
                # amplitude of sound should in [-1, 1]
                dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
            writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)

            writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

    writer.close()

.. currentmodule:: torch.utils.tensorboard.writer

.. autoclass:: SummaryWriter

   .. automethod:: add_scalar
   .. automethod:: add_histogram
   .. automethod:: add_image
   .. automethod:: add_figure
   .. automethod:: add_video
   .. automethod:: add_audio
   .. automethod:: add_text
   .. automethod:: add_graph
   .. automethod:: add_embedding
   .. automethod:: add_pr_curve
   .. automethod:: add_custom_scalars
