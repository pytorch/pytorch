torchvision.datasets
====================

The following dataset loaders are available:

-  `COCO (Captioning and Detection)`_
-  `LSUN Classification`_
-  `ImageFolder`_
-  `Imagenet-12`_
-  `CIFAR10 and CIFAR100`_

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
-  classes =
-  ‘train’ - all categories, training set
-  ‘val’ - all categories, validation set
-  ‘test’ - all categories, test set
-  [‘bedroom\_train’, ‘church\_train’, …] : a list of categories to load

CIFAR
~~~~~

``dset.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)``

``dset.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)``

-  ``root`` : root directory of dataset where there is folder
   ``cifar-10-batches-py``
-  ``train`` : ``True`` = Training set, ``False`` = Test set
-  ``download`` : ``True`` = downloads the dataset from the internet and
   puts it in root directory. If dataset already downloaded, do

.. _COCO (Captioning and Detection): #coco
.. _LSUN Classification: #lsun
.. _ImageFolder: #imagefolder
.. _Imagenet-12: #imagenet-12
.. _CIFAR10 and CIFAR100: #cifar
.. _COCO API to be installed: https://github.com/pdollar/coco/tree/master/PythonAPI
