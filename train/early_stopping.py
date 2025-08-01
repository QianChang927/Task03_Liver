import os
import torch

class EarlyStopping:
    def __init__(self, model, save_path, patience: int=None,
                 stop_criterion: str= 'valid', save_interval: int=10,
                 verbose: bool=False, train_compare=None, valid_compare=None):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.stop_criterion = stop_criterion
        self.save_interval = save_interval
        self.verbose = verbose

        self.train_compare = train_compare if train_compare else EarlyStoppingMethods.train_compare
        self.valid_compare = valid_compare if valid_compare else EarlyStoppingMethods.valid_compare

        self.counter = 0
        self.early_stop = False

        self.train_criteria = {}
        self.valid_criteria = {}

        self.best_criteria = -1
        self.best_epoch = -1

    def __call__(self, epoch, new_criteria, judge_mode: str):
        if judge_mode == 'train':
            compare = self.train_compare
        elif judge_mode == 'valid':
            compare = self.valid_compare
        else:
            raise ValueError('judge_mode must be "train" or "valid"')

        self.update(new_criteria, judge_mode)
        self.display(new_criteria, judge_mode)

        if (epoch + 1) % self.save_interval == 0:
            self.save_criteria(judge_mode)

        stop_flag = judge_mode == self.stop_criterion
        if not stop_flag: return

        if compare(self.best_criteria, new_criteria):
            self.best_criteria = new_criteria
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.patience and self.counter >= self.patience:
            self.early_stop = True

    def update(self, new_criteria: dict, update_mode: str):
        """
        用于更新字典内容
        :param new_criteria: 新字典
        :param update_mode: 更新模式
        """
        if update_mode == 'train':
            past_criteria = self.train_criteria
        elif update_mode == 'valid':
            past_criteria = self.valid_criteria
        else:
            raise ValueError('update_mode must be "train" or "valid"')

        for key, value in new_criteria.items():
            if key not in past_criteria:
                past_criteria[key] = []
            past_criteria[key].append(value)

    @staticmethod
    def display(criteria, prefix: str, concat: int=2):
        """
        输出字典内容
        :param criteria: 需要输出的内容
        :param prefix: 输出前缀
        :param concat: 一行输出concat个键值
        """
        cache = ''
        i = 0
        assert prefix in ['train', 'valid']
        for key, value in criteria.items():
            i += 1
            cache += f"{f'{prefix} {key}: {value:.5f}': <30}"
            if i % concat == 0:
                print(cache)
        if i % concat:
            print(cache)

    def save(self):
        if self.verbose:
            print(f"Saving model to {self.save_path}...")
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
        self.save_criteria('train')
        self.save_criteria('valid')

    def save_criteria(self, save_mode):
        if save_mode == 'train':
            criteria = self.train_criteria
        elif save_mode == 'valid':
            criteria = self.valid_criteria
        else:
            raise ValueError('save_mode must be "train" or "valid"')
        torch.save(criteria, os.path.join(self.save_path, f'{save_mode}_criteria.pt'))


class EarlyStoppingMethods:
    @staticmethod
    def train_compare(criteria_1: dict | float, criteria_2: dict | float) -> int:
        """
        比较train_criteria的优越度
        :param criteria_1: 源字典|浮点数
        :param criteria_2: 新字典|浮点数
        :return: 0-criteria_1更好，1-criteria_2更好，出错时默认返回criteria_1更好
        """
        cri_1 = criteria_1.get('loss', 0) if isinstance(criteria_1, dict) else criteria_1
        cri_2 = criteria_2.get('loss', 0) if isinstance(criteria_2, dict) else criteria_2
        return int(cri_1 > cri_2)

    @staticmethod
    def valid_compare(criteria_1: dict | float, criteria_2: dict | float) -> int:
        """
        比较valid_criteria的优越度
        :param criteria_1: 源字典|浮点数
        :param criteria_2: 新字典|浮点数
        :return: 0-criteria_1更好，1-criteria_2更好，出错时默认返回criteria_1更好
        """
        cri_1 = criteria_1.get('dice', 0) if isinstance(criteria_1, dict) else criteria_1
        cri_2 = criteria_2.get('dice', 0) if isinstance(criteria_2, dict) else criteria_2
        return int(cri_1 < cri_2)