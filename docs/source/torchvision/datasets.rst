torchvision.datasets
====================

The following dataset loaders are available:

-  `MNIST`_
-  `COCO (Captioning and Detection)`_
-  `LSUN Classification`_
-  `ImageFolder`_
-  `Imagenet-12`_
-  `CIFAR10 and CIFAR100`_
-  `STL10`_

Datasets have the API:

-  ``__getitem__``
-  ``__len__``
   They all subclass from ``torch.utils.data.Dataset``
   Hence, they can all be multi-threaded (python multiprocessing) using
   standard torch.utils.data.DataLoader.

For example:

``torch.utils.data.DataLoader(coco_cap, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)``

In the constructor, each dataset has a slightly different API as needed,
but they all take the keyword args:

-  ``transform`` - a function that takes in an image and returns a
   transformed version
-  common stuff like ``ToTensor``, ``RandomCrop``, etc. These can be
   composed together with ``transforms.Compose`` (see transforms section
   below)
-  ``target_transform`` - a function that takes in the target and
   transforms it. For example, take in the caption string and return a
   tensor of word indices.

MNIST
~~~~~

``dset.MNIST(root, train=True, transform=None, target_transform=None, download=False)``

- ``root`` : root directory of dataset where ``processed/training.pt`` and  ``processed/test.pt`` exist.
- ``train`` : ``True`` = Training set, ``False`` = Test set
-  ``download`` : ``True`` = downloads the dataset from the internet and puts it in root directory. If dataset already downloaded, place the processed dataset (function available in mnist.py) in the ``processed`` folder.

COCO
~~~~

This requires the `COCO API to be installed`_

Captions:
^^^^^^^^^

``dset.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])``

Example:

.. code:: python

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    cap = dset.CocoCaptions(root = 'dir where images are',
                            annFile = 'json annotation file',
                            transform=transforms.ToTensor())

    print('Number of samples: ', len(cap))
    img, target = cap[3] # load 4th sample

    print("Image Size: ", img.size())
    print(target)

Output:

::

    Number of samples: 82783
    Image Size: (3L, 427L, 640L)
    [u'A plane emitting smoke stream flying over a mountain.',
    u'A plane darts across a bright blue sky behind a mountain covered in snow',
    u'A plane leaves a contrail above the snowy mountain top.',
    u'A mountain that has a plane flying overheard in the distance.',
    u'A mountain view with a plume of smoke in the background']

Detection:
^^^^^^^^^^

``dset.CocoDetection(root="dir where images are", annFile="json annotation file", [transform, target_transform])``

LSUN
~~~~

``dset.LSUN(db_path, classes='train', [transform, target_transform])``

-  db\_path = root directory for the database files
-  ``classes`` = ``‘train’`` (all categories, training set), ``‘val’`` (all categories, validation set), ``‘test’`` (all categories, test set)
-  [``‘bedroom\_train’``, ``‘church\_train’``, …] : a list of categories to load

ImageFolder
~~~~~~~~~~~

A generic data loader where the images are arranged in this way:

::

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/asd932_.png

``dset.ImageFolder(root="root folder path", [transform, target_transform])``

It has the members:

-  ``self.classes`` - The class names as a list
-  ``self.class_to_idx`` - Corresponding class indices
-  ``self.imgs`` - The list of (image path, class-index) tuples

Imagenet-12
~~~~~~~~~~~

This is simply implemented with an ImageFolder dataset.

The data is preprocessed `as described
here <https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__

`Here is an
example <https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py#L48-L62>`__.

CIFAR
~~~~~

``dset.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)``

``dset.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)``

-  ``root`` : root directory of dataset where there is folder
   ``cifar-10-batches-py``
-  ``train`` : ``True`` = Training set, ``False`` = Test set
-  ``download`` : ``True`` = downloads the dataset from the internet and
   puts it in root directory. If dataset already downloaded, doesn't do anything.

STL10
~~~~~

``dset.STL10(root, split='train', transform=None, target_transform=None, download=False)``

-  ``root`` : root directory of dataset where there is folder ``stl10_binary``
-  ``split`` : ``'train'`` = Training set, ``'test'`` = Test set, ``'unlabeled'`` = Unlabeled set,    ``'train+unlabeled'`` = Training + Unlabeled set (missing label marked as ``-1``)
-  ``download`` : ``True`` = downloads the dataset from the internet and puts it in root directory. If dataset already downloaded, doesn't do anything.

.. _MNIST: #mnist
.. _COCO (Captioning and Detection): #coco
.. _LSUN Classification: #lsun
.. _ImageFolder: #imagefolder
.. _Imagenet-12: #imagenet-12
.. _CIFAR10 and CIFAR100: #cifar
.. _STL10: #stl10
.. _COCO API to be installed: https://github.com/pdollar/coco/tree/master/PythonAPI