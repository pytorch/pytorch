torch.utils.tensorboard
===================================

.. note::
    TensorBoard logging is used to visualize PyTorch models and metrics
    during their training process. Scalars, images, histograms, graphs, and
    embedding visualizations are supported for PyTorch models and tensors
    as well as Caffe2 nets and blobs.

    The SummaryWriter class is your main entry to log data for consumption
    and visualization by TensorBoard. For example:

    .. code:: python

        import torch
        import torchvision
        from torch.utils.tensorboard import SummaryWriter
        from torchvision import datasets, transforms

        writer = SummaryWriter()

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        model = torchvision.models.vgg16(False)
        images, labels = next(iter(trainloader))

        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)
        for n in range(100):
            writer.add_scalar('scalar1', torch.rand(1)[0], n)
        writer.close()

    This can then be visualized with TensorBoard, which should be installed
    with ``pip install tensorboard`` or equivalent. More details on TensorBoard
    can be found at https://www.tensorflow.org/tensorboard/

    .. warning::

        This code is EXPERIMENTAL and might change in the future. It also
        currently does not support all model types for ``add_graph``, which
        we are actively working on.

.. currentmodule:: torch.utils.tensorboard.writer
.. autoclass:: SummaryWriter
