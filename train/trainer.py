import os
import torch
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

# 类型注解所用库
from typing import Callable
from torch import device, Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from argparse import Namespace
from monai.data import DataLoader
from .early_stopping import EarlyStopping

class _Config:
    """
    用于保存常量的配置类
    """
    ROI_SIZE: tuple[int, int, int] = (64, 64, 64)
    SW_BATCH_SIZE: int = 2

class Trainer:
    def __init__(
        self,
        model: Module,
        loss_fn: Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        train_loader: DataLoader,
        valid_loader: DataLoader=None,
        save_dir: str=None,
        scheduler: LRScheduler=None,
        device: device=None,
        train_process: Callable[[Module, DataLoader, Callable[[dict, device], tuple[Tensor, Tensor]], Module, Optimizer, LRScheduler, device], dict|None]=None,
        valid_process: Callable[[Module, DataLoader, Callable[[dict, device], tuple[Tensor, Tensor]], Module, Optimizer, LRScheduler, device], dict|None]=None,
        batch_process: Callable[[dict, device], tuple[Tensor, Tensor]]=None,
        valid_interval: int=5,
        args: Namespace=None
    ) -> None:
        """
        训练器构造函数
        :param model: 所使用的神经网络
        :param loss_fn: 损失函数
        :param optimizer: 优化器
        :param early_stopping: 早停机制，包括进度输出功能
        :param train_loader: 所使用的训练数据集
        :param valid_loader: 所使用的验证数据集
        :param save_dir: 模型保存文件夹
        :param scheduler: 控制学习率变化的调度器
        :param device: 训练所用设备
        :param train_process: 训练过程函数：train_process(model, data_loader, batch_process, loss_fn, optimizer, scheduler, device) -> dict|None
        :param valid_process: 验证过程函数：valid_process(model, data_loader, batch_process, loss_fn, optimizer, scheduler, device) -> dict|None
        :param batch_process: batch处理函数：batch_process(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]
        :param valid_interval: 验证间隔
        :param args: 命令行参数解析器
        :return:
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if args is not None:
            _Config.ROI_SIZE = args.roi_size
            _Config.SW_BATCH_SIZE = args.sw_batch

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.train_process = train_process if train_process else TrainerMethods.train
        self.valid_process = valid_process if valid_process else TrainerMethods.valid
        self.batch_process = batch_process if batch_process else TrainerMethods.parse_batch
        self.valid_interval = valid_interval

    def run(self, epochs: int=100) -> None:
        """
        运行函数
        :param epochs: 训练轮次
        :return:
        """
        for epoch in range(epochs):
            print(f"{f'Epoch {epoch + 1}/{epochs}':-^60}")
            print(f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")

            self.model.train()
            train_criteria = self.train_process(self.model, self.train_loader, self.batch_process,
                                                self.loss_fn, self.optimizer, self.scheduler, self.device)
            if train_criteria is not None:
                self.early_stopping(epoch, train_criteria, 'train')

            if epoch % self.valid_interval != 0:
                continue

            self.model.eval()
            with torch.no_grad():
                valid_criteria = self.valid_process(self.model, self.valid_loader, self.batch_process,
                                                    self.loss_fn, self.optimizer, self.scheduler, self.device)
                if valid_criteria is not None:
                    self.early_stopping(epoch, valid_criteria, 'valid')

            if self.early_stopping.early_stop:
                print(f"{'':-^60}")
                print(f"Early stop: {epoch + 1}/{epochs}")
                self.early_stopping.end_display()
                break


class TrainerMethods:
    """
    训练器的默认静态方法类
    """
    @staticmethod
    def train(
        model: Module,
        data_loader: DataLoader,
        batch_process: Callable[[dict, device], tuple[Tensor, Tensor]],
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: device
    ) -> dict:
        """
        训练函数的默认实现
        :param model: 所使用的神经网络，若要在GPU上训练，应在调用此函数前转移
        :param data_loader: 所使用的训练数据集
        :param batch_process: batch解析函数，batch_process(batch, device)
        :param loss_fn: 损失函数，若要在GPU上训练，应在调用此函数前转移
        :param optimizer: 优化器
        :param scheduler: 控制动态学习率的调度器
        :param device: 训练所用设备
        :return: {'loss': epoch_loss}
        """
        train_step = 0
        epoch_loss = 0
        epoch_dice = 0

        def __calc_dice(y_pred: Tensor, y: Tensor) -> float:
            """
            计算dice矩阵
            :param y_pred: 预测值
            :param y: 真实值
            :return: 返回dice系数
            """
            import inspect
            from monai.losses import DiceLoss
            init_params = inspect.signature(DiceLoss).parameters
            valid_keys = set(init_params.keys())
            loss_fn_dict = loss_fn.__dict__
            filtered_dict = { key: value for key, value in loss_fn_dict.items() if key in valid_keys }
            dice_loss = DiceLoss(**filtered_dict)
            dice = 1 - dice_loss(y_pred, y).item()
            return dice

        for batch in data_loader:
            images, labels = batch_process(batch, device)
            train_step += 1

            _loss, _outputs = 0, 0
            def closure():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                nonlocal _loss, _outputs
                _loss, _outputs = loss.item(), outputs
                loss.backward()
                return loss

            optimizer.zero_grad()
            optimizer.step(closure)

            epoch_loss += _loss
            epoch_dice += __calc_dice(_outputs, labels)
            print(f"{train_step}/{len(data_loader)}, train loss: {_loss:.4f}")

        epoch_loss /= train_step
        epoch_dice /= train_step
        return {'loss': epoch_loss, 'dice': epoch_dice}

    @staticmethod
    def valid(
        model: Module,
        data_loader: DataLoader,
        batch_process: Callable[[dict, device], tuple[Tensor, Tensor]],
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: device
    ) -> dict:
        """
        验证函数的默认实现
        :param model: 所使用的神经网络，若要在GPU上训练，应在调用此函数前转移
        :param data_loader: 所使用的训练数据集
        :param batch_process: batch解析函数，batch_process(batch, device)
        :param loss_fn: 损失函数，若要在GPU上训练，应在调用此函数前转移
        :param optimizer: 优化器
        :param scheduler: 控制动态学习率的调度器
        :param device: 训练所用设备
        :return: {'dice': dice}
        """
        from monai.inferers import sliding_window_inference
        from monai.metrics import DiceMetric
        from monai.data import decollate_batch
        from monai import transforms

        dice_metric = DiceMetric(reduction='mean')
        post_pred = transforms.Compose([
            transforms.Activations(sigmoid=True),
            transforms.AsDiscrete(threshold=0.5)
        ])
        post_label = transforms.Compose([
            transforms.AsDiscrete()
        ])

        for batch in data_loader:
            images, labels = batch_process(batch, device)
            valid_outputs = sliding_window_inference(images, _Config.ROI_SIZE, _Config.SW_BATCH_SIZE, model)
            valid_outputs = [post_pred(i) for i in decollate_batch(valid_outputs)]
            valid_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=valid_outputs, y=valid_labels)

        dice = dice_metric.aggregate().item()
        dice_metric.reset()

        if scheduler is not None:
            scheduler.step(dice)

        return {'dice': dice}

    @staticmethod
    def parse_batch(batch: dict, device: device) -> tuple[Tensor, Tensor]:
        """
        解析batch的默认函数
        :param batch: 需要解析的batch
        :param device: 解析后的tensor数据存放在device上
        :return: 返回解析后的batch，该默认函数的返回类型为(Tensor, Tensor)
        """
        image, label = batch['image'], batch['label']
        label = label.int() & 1
        label = label.float()
        return image.to(device), label.to(device)
