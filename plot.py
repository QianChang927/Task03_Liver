import os
import math
import json
import torch

from collections.abc import Iterable
from matplotlib import pyplot as plt

OMIT_DICT = {
    'BatchNorm': 'BN',
    'InstanceNorm': 'IN'
}

class Drawer:
    def __init__(self, root_dir: str, modify_func=None, modify_args=None):
        self.root_dir = root_dir
        self.log_dirs = os.listdir(self.root_dir)
        if modify_func is not None:
            self.log_dirs = modify_func(self.root_dir, self.log_dirs, modify_args)
        self.log_legends = self.get_log_legends()

    def plot(self):
        if not len(self.log_dirs):
            raise IndexError('log_dirs should not be empty')

        for log_dir in self.log_dirs:
            self.draw_file(log_dir)
        plt.show()

    def get_log_legends(self) -> dict:
        if not len(self.log_dirs):
            return None

        revert_dict = {}
        # {
        #     'model': { 'UNet3D': ['file_1', 'file_2'], 'UNetMONAI': ['file_3'] }
        # }

        diff_dict = {}

        for log_dir in self.log_dirs:
            config = Drawer.get_config_json(os.path.join(self.root_dir, log_dir))
            for key, value in config.items():
                if key not in revert_dict:
                    revert_dict[key] = {}

                if not isinstance(value, str) and isinstance(value, Iterable):
                    value = '_'.join(map(str, value))

                if value not in revert_dict[key]:
                    revert_dict[key][value] = []

                revert_dict[key][value].append(log_dir)

        for key, value in revert_dict.items():
            if len(value.keys()) <= 1:
                continue

            for val, log_dirs in value.items():
                for log_dir in log_dirs:
                    if log_dir not in diff_dict:
                        diff_dict[log_dir] = {}
                    diff_dict[log_dir][key] = val

        if diff_dict == {}:
            for log_dir in self.log_dirs:
                diff_dict[log_dir] = log_dir

        else:
            for key, value in diff_dict.items():
                diff_dict[key] = Drawer.dict_to_str(diff_dict[key])

        revert_dict = {}
        for key, value in diff_dict.copy().items():
            if value not in revert_dict:
                revert_dict[value] = []

            revert_dict[value].append(key)
            if len(revert_dict[value]) > 1:
                diff_dict[key] = diff_dict[key] + f'_{key}'

        return diff_dict

    def get_drawing_layout(self, log_dir: str) -> tuple:
        key_words = {}
        file_dir = os.path.join(self.root_dir, log_dir)
        file_arr = os.listdir(file_dir)

        for file in file_arr:
            file_mode, file_content = Drawer.get_file_split(file)
            if file_content != 'criteria':
                continue

            file_dict = Drawer.get_file_dict(os.path.join(file_dir, file))
            for key in file_dict.keys():
                k = f'{file_mode}_{key}'
                if k not in key_words:
                    key_words[k] = 0

        return math.ceil(len(key_words) / 3), min(len(key_words), 3)

    def draw_file(self, log_dir: str) -> None:
        plt.figure('Criteria', (20, 6))
        row, col = self.get_drawing_layout(log_dir)

        subplot_index = 1
        subplot_dict = {}

        file_arr = os.listdir(os.path.join(self.root_dir, log_dir))
        file_dir = os.path.join(self.root_dir, log_dir)

        for file in file_arr:
            file_mode, file_content = Drawer.get_file_split(file)
            if file_content != 'criteria':
                continue

            file_dict = Drawer.get_file_dict(os.path.join(file_dir, file))
            for key, value in file_dict.items():
                subplot_title = f'{file_mode} {key}'
                if subplot_title not in subplot_dict:
                    subplot_dict[subplot_title] = subplot_index
                    subplot_index += 1

                index = subplot_dict[subplot_title]
                plt.subplot(row, col, index)

                plt.title(subplot_title)
                plt.xlabel('epoch')

                plt.plot([x + 1 for x in range(len(value))], value, label=self.log_legends[log_dir])
                plt.grid(True)
                plt.legend()

    @staticmethod
    def dict_to_str(ori_dict: dict) -> str:
        target_str = ''
        for key, value in ori_dict.items():
            if not isinstance(value, str) and isinstance(value, Iterable):
                value = '_'.join(map(str, value))
            target_str += f'{OMIT_DICT.get(key, key)}_{OMIT_DICT.get(value, value)}_'
        return target_str[:-1]

    @staticmethod
    def get_file_split(file: str, split: str = '_', ext: str = '.') -> tuple:
        file = file.split(ext)[0]
        file_split_arr = file.split(split)
        file_mode = file_split_arr[0].strip()
        file_content = file_split_arr[-1].strip()
        return file_mode, file_content

    @staticmethod
    def get_file_dict(file_path: str) -> dict:
        return torch.load(file_path)

    @staticmethod
    def get_config_json(dir_path: str) -> dict:
        file_arr = os.listdir(dir_path)
        if 'config.json' not in file_arr:
            raise FileExistsError('config.json not exists!')

        config = {}
        with open(os.path.join(dir_path, 'config.json'), 'r', encoding='UTF-8') as f:
            config = json.load(f)

        return config


class ModifyMethods:
    @staticmethod
    def omit_epoch_less(root_dir: str, log_dirs: list, args: list|None) -> list:
        threshold = args[0] if args else 500
        new_log_dirs = log_dirs.copy()

        for log_dir in log_dirs:
            config = Drawer.get_config_json(os.path.join(root_dir, log_dir))
            if config['epochs'] < threshold:
                new_log_dirs.remove(log_dir)

        return new_log_dirs

    @staticmethod
    def omit_layer(root_dir: str, log_dirs: list, args: list|None) -> list:
        layer = args[0] if args else 'BatchNorm'
        new_log_dirs = log_dirs.copy()

        for log_dir in log_dirs:
            config = Drawer.get_config_json(os.path.join(root_dir, log_dir))
            if config['layer'] == layer:
                new_log_dirs.remove(log_dir)

        return new_log_dirs

    @staticmethod
    def filter_lr(root_dir: str, log_dirs: list, args: list | None) -> list:
        lr = args if args else [1e-02]
        new_log_dirs = log_dirs.copy()

        for log_dir in log_dirs:
            config = Drawer.get_config_json(os.path.join(root_dir, log_dir))
            if config['lr'] not in lr:
                new_log_dirs.remove(log_dir)

        return new_log_dirs


if __name__ == '__main__':
    drawer = Drawer(
        root_dir='./checkpoint',
        modify_func=ModifyMethods.omit_epoch_less,
        modify_args=[100]
    )
    drawer.plot()