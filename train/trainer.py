import os
import torch
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
ROI_SIZE = (64, 64, 32)
SW_BATCH_SIZE = 4


class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader,
                 valid_loader=None, save_dir: str=None, scheduler=None, device=None,
                 train_process=None, valid_process=None, batch_process=None,
                 train_compare=None, valid_compare=None,
                 valid_interval=5) -> None:
        """
        初始化训练器
        :param model: 所使用的神经网络
        :param loss_fn: 损失函数
        :param optimizer: 优化器
        :param train_loader: 所使用的训练数据集
        :param valid_loader: 所使用的验证数据集
        :param save_dir: 模型保存文件夹
        :param scheduler: 控制学习率变化的调度器
        :param device: 训练所用设备
        :param train_process: 训练过程函数：train_process(model, data_loader, batch_process, loss_fn, optimizer,
              device, best_criteria, criteria_compare, save_dir) -> dict|None
        :param valid_process: 验证过程函数：valid_process(model, data_loader, batch_process, scheduler,
              device, best_criteria, criteria_compare, save_dir) -> dict|None
        :param batch_process: batch处理函数：batch_process(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]
        :param train_compare: train_criteria比较函数：train_compare(criteria_1: dict|float, criteria_2: dict|float) -> int
        :param valid_compare: valid_criteria比较函数：valid_compare(criteria_1: dict|float, criteria_2: dict|float) -> int
        :param valid_interval: 验证间隔
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.train_process = train_process if train_process else TrainerMethods.train
        self.valid_process = valid_process if valid_process else TrainerMethods.valid
        self.batch_process = batch_process if batch_process else TrainerMethods.parse_batch
        self.valid_interval = valid_interval

        self.train_compare = train_compare if train_compare else TrainerMethods.train_compare
        self.valid_compare = valid_compare if valid_compare else TrainerMethods.valid_compare

        self.train_criteria = {}
        self.valid_criteria = {}

        self.best_train_criteria = -1
        self.best_valid_criteria = -1

        self.best_train_epoch = -1
        self.best_valid_epoch = -1

    def run(self, epochs: int=100) -> None:
        """
        Trainer类的运行函数
        :param epochs: 训练轮次
        """
        def _update(src_dict: dict, new_dict: dict) -> None:
            """
            run的私有函数，用于更新字典内容
            :param src_dict: 源字典
            :param new_dict: 新字典
            """
            for key, value in new_dict.items():
                if key not in src_dict:
                    src_dict[key] = []
                src_dict[key].append(value)

        def _display(criteria: dict, prefix: str='train') -> None:
            """
            run的私有函数，用于输出字典内容
            :param criteria: 需要输出的字典，此处为评判标准
            :param prefix: 输出前缀
            """
            for key, value in criteria.items():
                print(f"{prefix} {key}: {value:.5f}")

        for epoch in range(epochs):
            print(f"{f'Epoch {epoch + 1}/{epochs}':-^50}")

            self.model.train()
            train_criteria = self.train_process(self.model, self.train_loader, self.batch_process,
                                                self.loss_fn, self.optimizer, self.device,
                                                self.best_train_criteria, self.train_compare, self.save_dir)
            if train_criteria is not None:
                _display(train_criteria, prefix='train')
                _update(self.train_criteria, train_criteria)
                if self.train_compare(self.best_train_criteria, train_criteria):
                    self.best_train_criteria = train_criteria
                    self.best_train_epoch = epoch

            if self.valid_criteria is None or epoch % self.valid_interval != 0:
                continue

            self.model.eval()
            with torch.no_grad():
                valid_criteria = self.valid_process(self.model, self.valid_loader, self.batch_process,
                                                    self.scheduler, self.device,
                                                    self.best_valid_criteria, self.valid_compare, self.save_dir)
                if valid_criteria is not None:
                    _display(valid_criteria, prefix='valid')
                    _update(self.valid_criteria, valid_criteria)
                    if self.valid_compare(self.best_valid_criteria, valid_criteria):
                        self.best_valid_criteria = valid_criteria
                        self.best_valid_epoch = epoch


class TrainerMethods:
    def __init__(self) -> None:
        pass

    @staticmethod
    def train(model, data_loader, batch_process, loss_fn, optimizer,
              device, best_criteria, criteria_compare, save_dir) -> dict:
        """
        训练函数的默认实现
        :param model: 所使用的神经网络，若要在GPU上训练，应在调用此函数前转移
        :param data_loader: 所使用的训练数据集
        :param batch_process: batch解析函数，batch_process(batch, device)
        :param loss_fn: 损失函数，若要在GPU上训练，应在调用此函数前转移
        :param optimizer: 优化器
        :param device: 训练所用设备
        :param best_criteria: 此前最佳评判表现，若要在训练过程中保存，此项为必要项
        :param criteria_compare: 评判指标比较函数，若要在训练过程中保存，此项为必要项
        :param save_dir: 模型保存位置，若要在训练过程中保存，该项为必要项
        :return: {'loss': epoch_loss}
        """
        train_step = 0
        epoch_loss = 0

        for batch in data_loader:
            images, labels = batch_process(batch, device)
            train_step += 1

            _loss = 0
            def closure():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                nonlocal _loss
                _loss = loss.item()
                loss.backward()
                return loss

            optimizer.zero_grad()
            optimizer.step(closure)

            epoch_loss += _loss
            print(f"{train_step}/{len(data_loader)}, train loss: {_loss:.4f}")

        epoch_loss /= train_step
        return {'loss': epoch_loss}

    @staticmethod
    def valid(model, data_loader, batch_process, scheduler,
              device, best_criteria, criteria_compare, save_dir) -> dict:
        """
        验证函数的默认实现
        :param model: 所使用的神经网络，若要在GPU上训练，应在调用此函数前转移
        :param data_loader: 所使用的验证数据集
        :param batch_process: batch解析函数，batch_process(batch, device)
        :param scheduler: 控制动态学习率的调度器
        :param device: 训练所用设备
        :param best_criteria: 此前最佳评判表现，若要在训练过程中保存，此项为必要项
        :param criteria_compare: 评判指标比较函数，若要在训练过程中保存，此项为必要项
        :param save_dir: 模型保存文件夹，若要在训练过程中保存，该项为必要项
        :return: {'dice': dice}
        """
        from monai.inferers import sliding_window_inference
        from monai.metrics import DiceMetric
        from monai.data import decollate_batch
        from monai import transforms
        import os

        dice_metric = DiceMetric(include_background=False, reduction='mean')
        post_pred = transforms.Compose([
            transforms.EnsureType(),
            transforms.AsDiscrete(argmax=True, to_onehot=model.out_channels)
        ])
        post_label = transforms.Compose([
            transforms.EnsureType(),
            transforms.AsDiscrete(to_onehot=model.out_channels)
        ])

        for batch in data_loader:
            images, labels = batch_process(batch, device)
            valid_outputs = sliding_window_inference(images, ROI_SIZE, SW_BATCH_SIZE, model)
            valid_outputs = [post_pred(i) for i in decollate_batch(valid_outputs)]
            valid_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=valid_outputs, y=valid_labels)

        dice = dice_metric.aggregate().item()
        dice_metric.reset()
        if scheduler is not None:
            scheduler.step(dice)

        if criteria_compare(best_criteria, dice) and save_dir is not None:
            torch.save(model.state_dict(), os.path.join(save_dir, 'valid_best_dice.pth'))

        return {'dice': dice}

    @staticmethod
    def parse_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        解析batch的默认函数
        :param batch: 需要解析的batch
        :param device: 解析后的tensor数据存放在device上
        :return: 返回解析后的batch，该默认函数的返回类型为(torch.Tensor, torch.Tensor)
        """
        image, label = batch['image'], batch['label']
        label = label.int() & 1
        label = label.float()
        return image.to(device), label.to(device)

    @staticmethod
    def train_compare(criteria_1: dict|float, criteria_2: dict|float) -> int:
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
    def valid_compare(criteria_1: dict|float, criteria_2: dict|float) -> int:
        """
        比较valid_criteria的优越度
        :param criteria_1: 源字典|浮点数
        :param criteria_2: 新字典|浮点数
        :return: 0-criteria_1更好，1-criteria_2更好，出错时默认返回criteria_1更好
        """
        cri_1 = criteria_1.get('dice', 0) if isinstance(criteria_1, dict) else criteria_1
        cri_2 = criteria_2.get('dice', 0) if isinstance(criteria_2, dict) else criteria_2
        return int(cri_1 < cri_2)