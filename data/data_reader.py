import os
import re
import sys
import glob
import random
import numpy as np

from monai import transforms
from monai.data import CacheDataset, DataLoader

SPATIAL_SIZE = (64, 64, 48)

class DataReader:
    def __init__(self, root_dir: str, train_dir: str, label_dir: str, test_dir: str,
                 data_transforms: dict=None, remain_nums: int = None,
                 val_scale: float=0.2, shuffle: bool=False,
                 num_workers=4, num_workers_loader=0) -> None:
        """
        实例化类，注意：DataReader只支持*.nii.gz后缀的文件
        :param root_dir: 数据集根文件夹
        :param train_dir: 训练集文件夹名
        :param label_dir: 标签文件夹名
        :param test_dir: 测试集文件夹名
        :param data_transforms: 以字典形式存放的transforms，{'train': monai.transforms, 'valid': monai.transforms, 'test': monai.transforms}
        :param remain_nums: 保留文件数量（训练集+验证集）
        :param val_scale: 验证集占数据集的百分比，范围(0, 1)
        :param shuffle: 是否打乱顺序
        :param num_workers: CacheDataset的加载线程数
        :param num_workers_loader: DataLoader的加载线程数
        """
        self.val_scale = val_scale
        if self.val_scale <= 0 or self.val_scale >= 1:
            raise ValueError('val_scale must be in (0, 1)')

        self.num_workers = num_workers
        self.num_workers_loader = num_workers_loader

        if sys.platform.startswith('win'):
            pattern = lambda x: int(re.findall(r'\d+', x)[0])
        else:
            pattern = None

        self.train_images = sorted(glob.glob(os.path.join(root_dir, train_dir, '*.nii.gz')), key=pattern)
        self.train_labels = sorted(glob.glob(os.path.join(root_dir, label_dir, '*.nii.gz')), key=pattern)
        self.test_images = sorted(glob.glob(os.path.join(root_dir, test_dir, '*.nii.gz')), key=pattern)

        self.train_valid_files = [{'image': image_name, 'label': label_name}
                                  for image_name, label_name in zip(self.train_images, self.train_labels)]

        self.data_files = {'test': [{'image': image_name} for image_name in self.test_images]}

        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.train_valid_files)
            random.shuffle(self.data_files['test'])

        if isinstance(remain_nums, int):
            self.set_data_nums(remain_nums)

        self.data_files['train'], self.data_files['valid'] = self.get_train_and_valid_file()
        self.data_cache = {}

        self.data_transforms = data_transforms if data_transforms is not None else {
            'train': transforms.Compose([
                transforms.LoadImaged(keys=['image', 'label']),
                transforms.EnsureChannelFirstd(keys=['image', 'label']),
                transforms.ScaleIntensityRanged(
                    keys=['image'],
                    a_min=-200,
                    a_max=200,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                transforms.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=True),
                transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                transforms.Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')),
                # transforms.Resized(keys=['image', 'label'], spatial_size=SPATIAL_SIZE),

                transforms.RandCropByPosNegLabeld(
                    keys=['image', 'label'],
                    image_key='image',
                    label_key='label',
                    spatial_size=SPATIAL_SIZE,
                    pos=1,
                    neg=1,
                    num_samples=4
                ),
                transforms.RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0,
                    spatial_size=SPATIAL_SIZE,
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)
                )
            ]),
            'valid': transforms.Compose([
                transforms.LoadImaged(keys=['image', 'label']),
                transforms.EnsureChannelFirstd(keys=['image', 'label']),
                transforms.ScaleIntensityRanged(
                    keys=['image'],
                    a_min=-200,
                    a_max=200,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                transforms.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=True),
                transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                transforms.Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')),
                # transforms.Resized(keys=['image', 'label'], spatial_size=SPATIAL_SIZE)
            ]),
            'test': transforms.Compose([
                transforms.LoadImaged(keys='image'),
                transforms.EnsureChannelFirstd(keys='image'),
                transforms.Orientationd(keys=['image'], axcodes='RAS'),
                transforms.Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2.0), mode='bilinear'),
                transforms.ScaleIntensityRanged(
                    keys=['image'],
                    a_min=-200,
                    a_max=200,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                transforms.CropForegroundd(keys=['image'], source_key='image', allow_smaller=True),
                # transforms.Resized(keys=['image', 'label'], spatial_size=SPATIAL_SIZE)
            ])
        }

    def get_train_and_valid_file(self) -> tuple:
        """
        获取train_files, valid_files
        :return: 分割后的训练集和验证集
        """
        num_files = int(len(self.train_valid_files) * self.val_scale)
        return self.train_valid_files[:-num_files], self.train_valid_files[-num_files:]

    def get_cache_dataset(self, target: str='train'):
        """
        生成DataCache，节约训练效率
        :param target: 需要生成的目标：['train', 'valid', 'test']
        """
        if target in self.data_cache:
            return

        if target not in self.data_transforms:
            raise ValueError("target must be in ['train', 'valid', 'test']")

        data_transforms = self.data_transforms[target]
        data_files = self.data_files[target]
        self.data_cache[target] = CacheDataset(data_files, data_transforms, num_workers=self.num_workers)

    def get_dataloader(self, target: str='train', batch_size: int=None) -> DataLoader:
        if target not in self.data_cache:
            self.get_cache_dataset(target=target)
        if batch_size is None:
            batch_size = 2 if target == 'train' else 1
        return DataLoader(self.data_cache[target], num_workers=self.num_workers_loader,
                          batch_size=batch_size, shuffle=self.shuffle)

    def set_data_nums(self, num: int) -> None:
        """
        仅保留data_dicts的前num项
        :param num: 要保留的数量
        """
        self.train_valid_files = self.train_valid_files[:num]


if __name__ == '__main__':
    data_reader = DataReader('../../Task_Dataset/Task03_Liver', 'imagesTr',
                             'labelsTr', 'imagesTs', remain_nums=10, val_scale=0.1, shuffle=True)
    train_loader = data_reader.get_dataloader(target='train')
    valid_loader = data_reader.get_dataloader(target='valid')