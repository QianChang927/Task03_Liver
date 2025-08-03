import os
import re
import sys
import json
import zlib

# 类型注解所用库
from typing import Protocol, Literal
from functools import partial
from re import Pattern
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from argparse import ArgumentParser
from data import DataReader
from train import Trainer, EarlyStopping

# 类型注解相关协议
class Stringfiable(Protocol):
    def __str__(self) -> str: ...

class Listifiable(Protocol):
    def tolist(self) -> list: ...

class InnerClass(Protocol):
    def __class__(self): ...

class ConfigParser:
    """
    配置文件解析类
    用于解析各种配置，并以json的形式保存为文件，便于后续复现及比对
    """

    def __init__(
            self,
            config_dir: str,
            args: ArgumentParser=None,
            data_reader: DataReader=None,
            model: Module=None,
            loss_fn: Module=None,
            optimizer: Optimizer=None,
            scheduler: LRScheduler=None,
            trainer: Trainer=None,
            early_stopping: EarlyStopping=None
    ) -> None:
        """
        类构造函数
        :param config_dir: 配置文件夹存放位置
        :param args: 运行参数
        :param data_reader: 数据读取器
        :param model: 网络模型
        :param loss_fn: 损失函数
        :param optimizer: 优化器
        :param scheduler: 学习率调度器
        :param trainer: 训练器
        :param early_stopping: 早停机制
        """
        self.config_dir = config_dir
        self.config_detail_dir = os.path.join(self.config_dir, 'config')
        os.makedirs(self.config_detail_dir, exist_ok=True)

        self.args = args
        self.data_reader = data_reader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainer = trainer
        self.early_stopping = early_stopping

        self.config_path = os.path.join(self.config_dir, 'config.json')
        self.config_dict = { 'system': sys.platform }

        self.model_dict = None
        self.transforms_dict = None
        self.loss_fn_dict = None
        self.optimizer_dict = None
        self.scheduler_dict = None
        self.trainer_dict = None
        self.early_stopping_dict = None

        def serialize(obj: type | Listifiable | Stringfiable) -> list | str:
            """
            处理JSON不可序列化的非基本数据类型
            :param obj: 需要序列化的对象
            :return: 转化后可序列化的对象
            """
            if type(obj).__name__ == 'type':
                return obj.__module__ + '.' + obj.__name__
            elif hasattr(obj, 'tolist') and callable(obj.tolist):
                obj_list = obj.tolist()
                return ['NaN' if isinstance(item, float) and item != item else item for item in obj_list]
            elif hasattr(obj, '__str__') and callable(obj.__str__):
                return str(obj)
            raise TypeError(f"Type {type(obj)} is not JSON serializable")

        def dict_to_hash(obj: dict) -> str:
            """
            字典转十六进制哈希值
            :param obj: 需要转为十六进制哈希值的字典
            :return: 转化后的十六进制哈希值
            """
            json_str = json.dumps(obj, sort_keys=True, default=serialize)
            crc32_hash = zlib.crc32(json_str.encode('utf-8'))
            return format(crc32_hash & 0xFFFFFFFF, '08X')

        def add_json_config(
                save_path: str,
                key_name: str,
                ori_dict: dict,
                add_to_config: bool=True
        ) -> None:
            """
            将字典形式的配置信息写入JSON文件
            :param save_path: 保存位置
            :param key_name: 添加在config.json中的键名
            :param ori_dict: 保存配置信息的字典
            :param add_to_config: 是否添加至config.json
            :return:
            """
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(ori_dict, f, indent=4, default=serialize)
            if add_to_config:
                hash_hex_str = dict_to_hash(ori_dict)
                self.config_dict.update({key_name: hash_hex_str})

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config_dict.update(json.load(f))

        if self.args is not None:
            self.parse_args()

        if self.data_reader is not None:
            self.data_transforms = self.data_reader.data_transforms
            for parse_mode in ['train', 'valid', 'test']:
                self.parse_data_reader(parse_mode)
            transforms_path = os.path.join(self.config_detail_dir, 'transforms.json')
            add_json_config(transforms_path, 'transforms', self.transforms_dict)

        if self.model is not None:
            self.parse_model()
            model_path = os.path.join(self.config_detail_dir, 'model.json')
            add_json_config(model_path, 'model', self.model_dict, False)

        if self.loss_fn is not None:
            self.parse_loss_function()
            loss_fn_path = os.path.join(self.config_detail_dir, 'loss_fn.json')
            add_json_config(loss_fn_path, 'loss_fn', self.loss_fn_dict)

        if self.optimizer is not None:
            self.parse_optimizer()
            optimizer_path = os.path.join(self.config_detail_dir, 'optimizer.json')
            add_json_config(optimizer_path, 'optimizer', self.optimizer_dict, False)

        if self.scheduler is not None:
            self.parse_scheduler()
            scheduler_path = os.path.join(self.config_detail_dir, 'scheduler.json')
            add_json_config(scheduler_path, 'scheduler', self.scheduler_dict)

        if self.trainer is not None:
            self.parse_trainer()
            trainer_path = os.path.join(self.config_detail_dir, 'trainer.json')
            add_json_config(trainer_path, 'trainer', self.trainer_dict, False)

        if self.early_stopping is not None:
            self.parse_early_stopping()
            early_stopping_path = os.path.join(self.config_detail_dir, 'early_stopping.json')
            add_json_config(early_stopping_path, 'early_stopping', self.early_stopping_dict, False)

        with open(self.config_path, 'w') as f:
            json.dump(self.config_dict, f, indent=4, default=serialize)

    def parse_args(self):
        """
        解析运行参数并保存配置信息
        :return:
        """
        self.config_dict.update(vars(self.args))

    def parse_data_reader(self, parse_mode: Literal['train', 'valid', 'test']) -> None:
        """
        解析DataReader中的transforms层级信息
        :param parse_mode: 解析模式
        :return:
        """
        def parse_transforms(tr_dict: dict) -> dict:
            """
            以递归方式解析transforms的层级信息
            :param tr_dict: 保存有transforms层级信息的字典
            :return:
            """
            key_remove = re.compile(r'^_.*', flags=re.DOTALL)
            val_remove = re.compile(r'.*RandomState.*|.*function.*at.*', flags=re.DOTALL)
            val_filter = re.compile(r'.*object at.*', flags=re.DOTALL)

            res_dict = {}
            for key, value in tr_dict.items():
                key_str = str(key)
                value_str = str(value)

                if bool(val_filter.match(value_str)):
                    try:
                        res_dict.update(parse_transforms(value.__dict__))
                    except AttributeError:
                        pass

                elif bool(key_remove.match(key_str)):
                    continue

                elif bool(val_remove.match(value_str)):
                    continue

                else:
                    res_dict.update({key: 'NaN' if isinstance(value, float) and value != value else value})
            return res_dict

        if self.transforms_dict is None:
            self.transforms_dict = {}
        self.transforms_dict[parse_mode] = []
        data_transforms = self.data_transforms[parse_mode].transforms
        for transforms in data_transforms:
            self.transforms_dict[parse_mode].append({
                type(transforms).__name__: parse_transforms(transforms.__dict__)
            })
    
    def parse_model(self) -> None:
        """
        解析模型的参数信息
        :return:
        """
        if self.model_dict is None:
            self.model_dict = {}
        self.model_dict[self.model.__class__.__name__] = str(self.model)

    def parse_loss_function(self) -> None:
        """
        解析损失函数的参数信息
        :return:
        """
        if self.loss_fn_dict is None:
            self.loss_fn_dict = {}
        key_remove = re.compile(r'^_.*', flags=re.DOTALL)
        loss_fn_name = ConfigParser.get_obj_name(self.loss_fn)
        loss_fn_dict = self.loss_fn.__dict__
        self.loss_fn_dict[loss_fn_name] = {}
        ConfigParser.parse_process(loss_fn_dict, self.loss_fn_dict[loss_fn_name], key_remove)

    def parse_optimizer(self) -> None:
        """
        解析优化器的参数信息
        :return:
        """
        if self.optimizer_dict is None:
            self.optimizer_dict = {}
        key_remove = re.compile(r'^_.*|.*param_groups.*', flags=re.DOTALL)
        optimizer_name = ConfigParser.get_obj_name(self.optimizer)
        optimizer_dict = self.optimizer.__dict__
        self.optimizer_dict[optimizer_name] = {}
        ConfigParser.parse_process(optimizer_dict, self.optimizer_dict[optimizer_name], key_remove)

    def parse_scheduler(self) -> None:
        """
        解析学习率调度器的参数信息
        :return:
        """
        if self.scheduler_dict is None:
            self.scheduler_dict = {}
        key_remove = re.compile(
            r'^_.*|.*optimizer.*|.*mode_worse.*|.*best.*|.*min_lrs.*|.*last_epoch.*|.*num_bad_epochs.*|.*cooldown_counter.*',
            flags=re.DOTALL
        )
        scheduler_name = ConfigParser.get_obj_name(self.scheduler)
        scheduler_dict = self.scheduler.__dict__
        self.scheduler_dict[scheduler_name] = {}
        ConfigParser.parse_process(scheduler_dict, self.scheduler_dict[scheduler_name], key_remove)

    def parse_trainer(self) -> None:
        """
        解析训练器的参数信息
        :return:
        """
        if self.trainer_dict is None:
            self.trainer_dict = {}
        key_remove = re.compile(r'^_.*|.*model.*|.*loss_fn.*|.*optimizer.*', flags=re.DOTALL)
        val_remove = re.compile(r'.*object at.*|.*function.*at.*', flags=re.DOTALL)
        trainer_name = ConfigParser.get_obj_name(self.trainer)
        trainer_dict = self.trainer.__dict__
        self.trainer_dict[trainer_name] = {}
        ConfigParser.parse_process(trainer_dict, self.trainer_dict[trainer_name], key_remove, val_remove)

    def parse_early_stopping(self) -> None:
        """
        解析早停机制的参数信息
        :return:
        """
        if self.early_stopping_dict is None:
            self.early_stopping_dict = {}
        key_remove = re.compile(r'^_.*|.*compare.*|.*model.*|.*best.*|.*criteria.*', flags=re.DOTALL)
        early_stopping_name = ConfigParser.get_obj_name(self.early_stopping)
        early_stopping_dict = self.early_stopping.__dict__
        self.early_stopping_dict[early_stopping_name] = {}
        ConfigParser.parse_process(early_stopping_dict, self.early_stopping_dict[early_stopping_name], key_remove)

    @staticmethod
    def get_obj_name(obj: partial | InnerClass) -> str:
        """
        获取类名
        :param obj: 需要获取名字的类
        :return: 类名
        """
        if isinstance(obj, partial):
            ori_class = obj.func
            return ori_class.__name__
        else:
            return obj.__class__.__name__

    @staticmethod
    def parse_process(
            parse_dict: dict,
            target_dict: dict,
            key_remove: Pattern=None,
            val_remove: Pattern=None
    ) -> None:
        """
        根据re.Pattern从parse_dict筛选符合条件的键值对并保存至target_dict
        :param parse_dict: 需要解析的字典
        :param target_dict: 负责保存数据的字典
        :param key_remove: 键删除模板
        :param val_remove: 值删除模板
        :return:
        """
        for key, value in parse_dict.items():
            key_str = str(key)
            val_str = str(value)
            if key_remove and bool(key_remove.match(key_str)):
                continue
            if val_remove and bool(val_remove.match(val_str)):
                continue
            target_dict.update({key: value})