import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import webdataset as wds
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class HFImageDataset(Dataset):
    def __init__(self, name, split, transforms, image_key, label_key='label'):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError('Please install datasets with `pip install datasets`.')

        self.df = load_dataset(name, split=split)
        self.size = len(self.df)
        self.image_key = image_key
        self.label_key = label_key
        self.transforms = transforms

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self.transforms(self.df[idx][self.image_key])
        return {'image': image, 'label': self.df[idx][self.label_key]}


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
    image_size: int,
    is_train: bool,
    mean=None,
    std=None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        train_transform = Compose([
            RandomResizedCrop(
                image_size,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return train_transform
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def default_wds_collate(examples):
    images = torch.stack([example['image'] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return images, labels


def get_wds(*, data, preprocess_fn, batch_size, num_workers):
    expanded_urls = wds.shardlists.expand_urls(data)

    pipeline = [wds.SimpleShardList(expanded_urls)]
    pipeline.extend([
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ])
    pipeline.extend([
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", label='cls'),
        wds.map_dict(image=preprocess_fn),
        wds.batched(
            batch_size,
            partial=False,
            collation_fn=None,
        ),
    ])

    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        collate_fn=default_wds_collate,
    )
    return dataloader


def get_hf_image_dataset(*, data, preprocess_fn, batch_size, num_workers, image_key):
    dataset_name = data
    assert dataset_name

    dataset = HFImageDataset(
        dataset_name,
        split='train',
        transforms=preprocess_fn,
        image_key=image_key,
    )
    num_samples = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader
