import os
import re
import sys
import json
import zlib

class ConfigParser:
    def __init__(self, config_dir, args=None, data_reader=None, model=None,
                 loss_fn=None, optimizer=None, scheduler=None):
        self.config_dir = config_dir
        self.config_detail_dir = os.path.join(self.config_dir, 'config')
        os.makedirs(self.config_detail_dir, exist_ok=True)

        self.args = args
        self.data_reader = data_reader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config_path = os.path.join(self.config_dir, 'config.json')
        self.config_dict = { 'system': sys.platform }

        self.model_dict = None
        self.transforms_dict = None
        self.loss_fn_dict = None
        self.optimizer_dict = None
        self.scheduler_dict = None

        def serialize(obj):
            if type(obj).__name__ == 'type':
                return obj.__module__ + '.' + obj.__name__
            elif hasattr(obj, 'tolist') and callable(obj.tolist):
                obj_list = obj.tolist()
                return ['NaN' if isinstance(item, float) and item != item else item for item in obj_list]
            elif hasattr(obj, '__str__') and callable(obj.__str__):
                return str(obj)
            raise TypeError(f"Type {type(obj)} is not JSON serializable")

        def dict_to_hash(obj):
            json_str = json.dumps(obj, sort_keys=True, default=serialize)
            crc32_hash = zlib.crc32(json_str.encode('utf-8'))
            return format(crc32_hash & 0xFFFFFFFF, '08X')

        def add_json_config(save_path, key_name, ori_dict, add_to_config=True):
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

        with open(self.config_path, 'w') as f:
            json.dump(self.config_dict, f, indent=4, default=serialize)

    def parse_args(self):
        self.config_dict.update(vars(self.args))

    def parse_data_reader(self, parse_mode):
        def parse_transforms(tr_dict):
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
    
    def parse_model(self):
        if self.model_dict is None:
            self.model_dict = {}
        self.model_dict[self.model.__class__.__name__] = str(self.model)

    def parse_loss_function(self):
        if self.loss_fn_dict is None:
            self.loss_fn_dict = {}
        key_remove = re.compile(r'^_.*', flags=re.DOTALL)
        loss_fn_name = ConfigParser.get_obj_name(self.loss_fn)
        loss_fn_dict = self.loss_fn.__dict__
        self.loss_fn_dict[loss_fn_name] = {}
        ConfigParser.parse_process(loss_fn_dict, self.loss_fn_dict[loss_fn_name], key_remove)

    def parse_optimizer(self):
        if self.optimizer_dict is None:
            self.optimizer_dict = {}
        key_remove = re.compile(r'^_.*|.*param_groups.*', flags=re.DOTALL)
        optimizer_name = ConfigParser.get_obj_name(self.optimizer)
        optimizer_dict = self.optimizer.__dict__
        self.optimizer_dict[optimizer_name] = {}
        ConfigParser.parse_process(optimizer_dict, self.optimizer_dict[optimizer_name], key_remove)

    def parse_scheduler(self):
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

    @staticmethod
    def get_obj_name(obj):
        from functools import partial
        if isinstance(obj, partial):
            ori_class = obj.func
            return ori_class.__name__
        else:
            return obj.__class__.__name__

    @staticmethod
    def parse_process(parse_dict, target_dict, key_remove=None, val_remove=None):
        for key, value in parse_dict.items():
            key_str = str(key)
            val_str = str(value)
            if key_remove and bool(key_remove.match(key_str)):
                continue
            if val_remove and bool(val_remove.match(val_str)):
                continue
            target_dict.update({key: value})