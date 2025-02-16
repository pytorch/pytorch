# Adapted from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py
import os
from pathlib import Path
from typing import Any, List, Union, Callable, Optional, TypeVar
from typing_extensions import ParamSpec


import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import LightningDataModule

from torchvision import transforms
from torchvision.datasets import ImageFolder



P = ParamSpec('P')
T = TypeVar('T')



class DictDataset(Dataset):

    def __init__(self, dataset_dict, length=None):
        """dataset_dict: dictionary mapping from index to batch
        length is used in the case of DistributedSampler: e.g. the dataset could have size 1k, but
        with 8 GPUs the dataset_dict would only have 125 items.
        """
        super().__init__()
        self.dataset_dict = dataset_dict
        self.length = length or len(self.dataset_dict)

    def __getitem__(self, index):
        return self.dataset_dict[index]

    def __len__(self):
        return self.length


# From https://github.com/PyTorchLightning/lightning-bolts/blob/2415b49a2b405693cd499e09162c89f807abbdc4/pl_bolts/transforms/dataset_normalizations.py#L10
def imagenet_normalization():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ImagenetDataModule(LightningDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/
        Sample-of-Images-from-the-ImageNet-Dataset-used-in-the-ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet
    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)
    Imagenet train, val and test dataloaders.
    The train set is the imagenet train.
    The val set is taken from the train set with `num_imgs_per_val_class` images per class.
    For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.
    The test set is the official imagenet validation set.
     Example::
        from pl_bolts.datamodules import ImagenetDataModule
        dm = ImagenetDataModule(IMAGENET_PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "imagenet"

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        img_dtype='float32',  # Using str since OmegaConf doesn't support non-primitive type
        cache_val_dataset=False,
        mixup: Optional[Callable] = None,
        num_aug_repeats: int = 0,
        num_workers: int = 0,
        batch_size: int = 32,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            num_imgs_per_val_class: how many images per class for the validation set
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        assert img_dtype in ['float32', 'float16', 'bfloat16']
        self.img_dtype = torch.__getattribute__(img_dtype)
        self.cache_val_dataset = cache_val_dataset
        self.mixup = mixup
        self.num_aug_repeats = num_aug_repeats
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = Path(data_dir).expanduser()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.
        .. warning:: Please download imagenet on your own first.
        """
        self._verify_splits(self.data_dir, "train")
        self._verify_splits(self.data_dir, "val")

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = (self.train_transform() if self.train_transforms is None
                                else self.train_transforms)
            val_transforms = (self.val_transform() if self.val_transforms is None
                              else self.val_transforms)
            if self.img_dtype is not torch.float32:
                assert isinstance(train_transforms, transforms.Compose)
                assert isinstance(val_transforms, transforms.Compose)
                convert_dtype = transforms.Lambda(lambda x: x.to(dtype=self.img_dtype))
                train_transforms.transforms.append(convert_dtype)
                val_transforms.transforms.append(convert_dtype)
            self.dataset_train = ImageFolder(self.data_dir / 'train', transform=train_transforms)
            self.dataset_val = ImageFolder(self.data_dir / 'val', transform=val_transforms)

        if stage == "test" or stage is None:
            test_transforms = (self.val_transform() if self.test_transforms is None
                               else self.test_transforms)
            if self.img_dtype is not torch.float32:
                assert isinstance(test_transforms, transforms.Compose)
                convert_dtype = transforms.Lambda(lambda x: x.to(dtype=self.img_dtype))
                test_transforms.transforms.append(convert_dtype)
            self.dataset_test = ImageFolder(self.data_dir / 'val', transform=test_transforms)

    def train_transform(self) -> Callable:
        """The standard imagenet transforms.
        .. code-block:: python
            transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.
        .. code-block:: python
            transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing

    def train_dataloader(self, *args: P.args, **kwargs: P.kwargs) -> DataLoader:
        """ The train dataloader """
        if self.num_aug_repeats == 0:
            shuffle = self.shuffle
            sampler = None
        else:
            shuffle = False
            from timm.data.distributed_sampler import RepeatAugSampler
            sampler = RepeatAugSampler(self.dataset_train, num_repeats=self.num_aug_repeats)
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, mixup=self.mixup, sampler=sampler)

    def val_dataloader(self, *args: P.args, **kwargs: P.kwargs) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        # If using RepeatAugment, we set trainer.replace_sampler_ddp=False, so we have to
        # construct the DistributedSampler ourselves.
        if not self.cache_val_dataset:
            sampler = (DistributedSampler(self.dataset_val, shuffle=False, drop_last=self.drop_last)
                       if self.num_aug_repeats != 0 else None)
            return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval,
                                     sampler=sampler)
        else:
            print('Caching val dataset')
            sampler = (SequentialSampler(self.dataset_val) if self.trainer.world_size <= 1
                       else DistributedSampler(self.dataset_val, shuffle=False,
                                               drop_last=self.drop_last))
            indices = list(iter(sampler))
            loader = DataLoader(self.dataset_val, batch_size=None, shuffle=False, sampler=sampler,
                                num_workers=self.num_workers, drop_last=self.drop_last)
            batches = list(loader)
            assert len(batches) == len(indices)
            self.dataset_val = DictDataset(dict(zip(indices, batches)),
                                           length=len(self.dataset_val))
            sampler = (DistributedSampler(self.dataset_val, shuffle=False, drop_last=self.drop_last)
                       if self.num_aug_repeats != 0 else None)
            return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval,
                                     sampler=sampler)

    def test_dataloader(self, *args: P.args, **kwargs: P.kwargs) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        sampler = (DistributedSampler(self.dataset_test, shuffle=False, drop_last=self.drop_last)
                   if self.num_aug_repeats != 0 else None)
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval, sampler=sampler)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     mixup: Optional[Callable] = None, sampler=None) -> DataLoader:
        collate_fn = ((lambda batch: mixup(*default_collate(batch))) if mixup is not None
                      else default_collate)
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
    

class Imagenet21kPDataModule(ImagenetDataModule):
    """ImageNet-21k (winter 21) processed with https://github.com/Alibaba-MIIL/ImageNet21K
    """

    @property
    def num_classes(self) -> int:
        """
        Return:
            10450
        """
        return 10450
