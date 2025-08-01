import os
import math
import json
import torch

from collections.abc import Iterable, Callable
from matplotlib import pyplot as plt
from datetime import datetime

OMIT_DICT = {
    'BatchNorm': 'BN',
    'InstanceNorm': 'IN'
}

class Drawer:
    def __init__(self, root_dir: str, modify_process: list|dict=None):
        self.root_dir = root_dir
        self.log_dirs = os.listdir(self.root_dir)
        if modify_process is not None:
            if not isinstance(modify_process, list):
                modify_process = [modify_process]
            for modify_dict in modify_process:
                modify_func: Callable = modify_dict.get('func', None)
                modify_args: Iterable = modify_dict.get('args', None)
                if modify_func is None: continue
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

        '''
        思路：
        1.  log_dir1 <=> { 'attr1': 'value1', 'attr2': 'value2' } -> { 'attr1': { 'value1': ['log_dir1', ...] }, 'attr2': { 'value2': ['log_dir1', ...] } }
        2.  统计attr*中键的个数
            1)  若不唯一，则differ_dict[log_dir*]['attr*'] = 'value*'
            2)  若唯一，则differ_dict忽略attr*
        3.  统计attr*中['log_dir1', ...]的总长度，比较其与len(log_dirs)的大小
            1)  若不一致，则differ_dict[log_dir*]['attr*'] = 'value*'
            2)  若一致，则differ_dict忽略attr*
        4.  differ_dict[log_dir*] = dict_to_str(differ_dict[log_dir*])
        '''

        revert_dict = {}
        differ_dict = {}

        # 生成反置键值对
        for log_dir in self.log_dirs:
            config_dict = Drawer.get_config_json(os.path.join(self.root_dir, log_dir))
            for key, value in config_dict.items():
                # 忽略输入和输出文件夹设置
                if key in ['input', 'output']:
                    continue
                if not isinstance(value, str) and isinstance(value, Iterable):
                    value = '-'.join(map(str, value))
                if key not in revert_dict:
                    revert_dict[key] = {}
                if value not in revert_dict[key]:
                    revert_dict[key][value] = []
                revert_dict[key][value].append(log_dir)

        # 检测反置键值对的长度
        for key, value in revert_dict.items():
            if len(value.keys()) == 1:
                log_dir_length = 0
                for v, log_dirs in value.items():
                    log_dir_length += len(log_dirs)

                if log_dir_length == len(self.log_dirs):
                    continue

            for v, log_dirs in value.items():
                for log_dir in log_dirs:
                    if log_dir not in differ_dict:
                        differ_dict[log_dir] = {}
                    differ_dict[log_dir][key] = v

        # 将differ_dict内部的键值对转为字符串
        for key, value in differ_dict.items():
            differ_dict[key] = Drawer.dict_to_str(differ_dict[key])

        # 防止不同log_dir的differ_dict[log_dir]完全一致
        revert_dict = {}
        for key, value in differ_dict.items():
            if value not in revert_dict:
                revert_dict[value] = []

            revert_dict[value].append(key)
            if len(revert_dict[value]) > 1:
                differ_dict[key] += f'_FILE_{key}'

        return differ_dict

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
        if not isinstance(ori_dict, dict):
            return str(ori_dict)

        target_str = ''
        for key, value in ori_dict.items():
            if not isinstance(value, str) and isinstance(value, Iterable):
                value = '-'.join(map(str, value))
            target_str += f'{OMIT_DICT.get(key, key.upper())}_{OMIT_DICT.get(value, value)}_'
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
    def filter_kwargs(root_dir: str, log_dirs: list, args: list) -> list:
        """
        筛选config中的kwargs，保留/丢弃符合条件的log_dir
        :param root_dir:
        :param log_dirs:
        :param args: 此参数为空时直接返回log_dirs，[(Optional)mode['omit', 'select'], {key1: value1, key2: value2, ...}, {key3: value3, ...}, ...]
        :return: new_log_dirs
        """

        if not args or not isinstance(args, list):
            return log_dirs

        if isinstance(args[0], str):
            mode = args.pop(0).strip().lower()
        else:
            mode = 'select'   # omit: 省略, select: 选取

        if len(args) < 1:
            return log_dirs

        if mode == 'omit':
            filter_init = False
            filter_lambda = lambda x, y: x or y
            new_log_dirs = log_dirs.copy()
        elif mode == 'select':
            filter_init = True
            filter_lambda = lambda x, y: x and y
            new_log_dirs = []
        else:
            raise ValueError('`mode` should be `omit` or `select`')

        def _modify(target_list, element):
            if mode == 'omit':
                target_list.remove(element)
            else:
                target_list.append(element)

        def _judge(_flag) -> bool:
            if mode == 'omit':
                return _flag
            else:
                return not _flag

        for log_dir in log_dirs:
            config = Drawer.get_config_json(os.path.join(root_dir, log_dir))

            flag = filter_init
            for kwargs in args:
                if _judge(flag): break
                for key, value in kwargs.items():
                    if _judge(flag): break
                    if isinstance(value, str) or not isinstance(value, Iterable):
                        value = [value]
                    flag = filter_lambda(flag, config.get(key, None) in value)

            if flag:
                _modify(new_log_dirs, log_dir)

        return new_log_dirs

    @staticmethod
    def filter_ctime(root_dir: str, log_dirs: list, args: list) -> list:
        """
        筛选文件创建时间，保留/丢弃符合条件的log_dir
        :param root_dir:
        :param log_dirs:
        :param args: 此参数为空时直接返回log_dirs，[(Optional)mode['omit', 'select'], time_start, time_end]
        :return: new_log_dirs
        """

        if not args or not isinstance(args, list):
            return log_dirs

        if isinstance(args[0], str):
            mode = args.pop(0).strip().lower()
        else:
            mode = 'select'  # omit: 省略, select: 选取

        if len(args) < 2:
            return log_dirs

        if mode == 'omit':
            new_log_dirs = log_dirs.copy()
        elif mode == 'select':
            new_log_dirs = []
        else:
            raise ValueError('`mode` should be `omit` or `select`')

        time_start = args.pop(0)
        time_end = args.pop(0)

        def _check(file_ctime):
            return time_start <= file_ctime < time_end

        def _modify(target_list, element):
            if mode == 'omit':
                target_list.remove(element)
            else:
                target_list.append(element)

        def _get_ctime(file_path):
            create_time = os.path.getctime(file_path)
            return datetime.fromtimestamp(create_time)

        for log_dir in log_dirs:
            file_path = os.path.join(root_dir, log_dir)
            file_ctime = _get_ctime(file_path)
            if _check(file_ctime):
                _modify(new_log_dirs, log_dir)

        return new_log_dirs


if __name__ == '__main__':
    drawer = Drawer(
        root_dir='./checkpoint',
        modify_process=[
            { 'func': ModifyMethods.filter_ctime, 'args': ['select', datetime(2025, 7, 29), datetime(2025, 7, 30)] },
            { 'func': ModifyMethods.filter_kwargs, 'args': ['select', {'system': 'linux'}] }
        ]
    )
    drawer.plot()