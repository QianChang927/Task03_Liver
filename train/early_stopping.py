import os
import torch

# 类型注解所用库
from torch.nn import Module
from typing import Literal, Callable

class EarlyStopping:
    """
    早停机制实现类，包含早停机制、进度输出及进度保存
    """
    def __init__(
        self,
        model: Module,
        save_dir: str,
        patience: int=None,
        min_delta: float=0.,
        stop_criterion: Literal['train', 'valid']= 'valid',
        save_interval: int=10,
        verbose: bool=False,
        train_compare: Callable[[dict | float, dict | float], float]=None,
        valid_compare: Callable[[dict | float, dict | float], float]=None
    ) -> None:
        """
        早停机制类构造函数
        :param model: 实例化后的模型类
        :param save_dir: 文件保存位置
        :param patience: 连续patience个epoch满足条件后停止，为None时表示不需要早停
        :param min_delta: 模型优化的最小阈值
        :param stop_criterion: 早停评估标准，以训练还是验证的数据为评判标准
        :param save_interval: 进度保存间隔
        :param verbose: 是否输出额外信息
        :param train_compare: 训练过程比较函数，输出优越差
        :param valid_compare: 验证过程比较函数，输出优越差
        :return:
        """
        self.model = model
        self.save_dir = save_dir
        self.patience = patience
        self.min_delta = min_delta
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

    def __call__(self, epoch: int, new_criteria: dict, judge_mode: Literal['train', 'valid']) -> None:
        """
        类调用方法
        :param epoch: 当前轮次
        :param new_criteria: 当前评判指标
        :param judge_mode: 评判模式
        :return:
        """
        def __display(criteria: dict, concat: int = 2) -> None:
            """
            输出字典内容
            :param criteria: 需要输出的内容
            :param concat: 一行输出concat个键值
            :return:
            """
            cache = ''
            i = 0
            for key, value in criteria.items():
                i += 1
                cache += f"{f'{judge_mode} {key}: {value:.5f}': <30}"
                if i % concat == 0:
                    print(cache)
            if i % concat:
                print(cache)

        def __save():
            """
            更新最佳评判标准并保存模型及评判标准变化曲线
            :return:
            """
            self.best_criteria = new_criteria
            self.best_epoch = epoch
            self.__save_model()
            self.__save_criteria(judge_mode)

        if judge_mode == 'train':
            compare = self.train_compare
        elif judge_mode == 'valid':
            compare = self.valid_compare
        else:
            raise ValueError('judge_mode must be "train" or "valid"')

        self.__update(new_criteria, judge_mode)
        __display(new_criteria)

        if (epoch + 1) % self.save_interval == 0:
            self.__save_criteria(judge_mode)

        stop_flag = judge_mode == self.stop_criterion
        if not stop_flag: return

        if compare_result := compare(self.best_criteria, new_criteria) > self.min_delta:    # 模型有明显改进
            __save()
            self.counter = 0
        elif compare_result > 0:                                                            # 模型仅有略微进步
            __save()
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def end_display(self) -> None:
        """
        结束信息输出
        :return:
        """
        print(f"best criteria: {self.best_criteria} at epoch {self.best_epoch + 1}")

    def __update(self, new_criteria: dict, update_mode: Literal['train', 'valid']) -> None:
        """
        用于更新字典内容
        :param new_criteria: 新字典
        :param update_mode: 更新模式
        :return:
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

    def __save_model(self) -> None:
        """
        保存模型
        :return:
        """
        if self.verbose: print(f"Saving model to {self.save_dir}...")
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))

    def __save_criteria(self, save_mode: Literal['train', 'valid']) -> None:
        """
        保存评判标准变化曲线
        :param save_mode: 保存模式
        :return:
        """
        if save_mode == 'train':
            criteria = self.train_criteria
        elif save_mode == 'valid':
            criteria = self.valid_criteria
        else:
            raise ValueError('save_mode must be "train" or "valid"')
        torch.save(criteria, os.path.join(self.save_dir, f'{save_mode}_criteria.pt'))


class EarlyStoppingMethods:
    """
    早停机制默认静态方法类
    """
    @staticmethod
    def train_compare(criteria_1: dict | float, criteria_2: dict | float) -> float:
        """
        训练标准优越度计算：返回criteria_1与criteria_2的优越差值
        :param criteria_1: 源字典|浮点数
        :param criteria_2: 新字典|浮点数
        :return: >0 criteria_2更优越; <0 criteria_1更优越; =0 二者无区别
        """
        cri_1 = criteria_1.get('loss', 0.) if isinstance(criteria_1, dict) else criteria_1
        cri_2 = criteria_2.get('loss', 0.) if isinstance(criteria_2, dict) else criteria_2
        return float(cri_1 - cri_2)

    @staticmethod
    def valid_compare(criteria_1: dict | float, criteria_2: dict | float) -> float:
        """
        验证标准优越度计算：返回criteria_1与criteria_2的优越差值
        :param criteria_1: 源字典|浮点数
        :param criteria_2: 新字典|浮点数
        :return: >0 criteria_2更优越; <0 criteria_1更优越; =0 二者无区别
        """
        cri_1 = criteria_1.get('dice', 0.) if isinstance(criteria_1, dict) else criteria_1
        cri_2 = criteria_2.get('dice', 0.) if isinstance(criteria_2, dict) else criteria_2
        return float(cri_2 - cri_1)