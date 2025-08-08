import os
import torch
import monai.networks.nets as monai_nets
import monai.networks.layers as monai_layers

from data import DataReaderMSD
from model import UNet3D, VNet3D
from train import Trainer, EarlyStopping
from parser import ArgParser, ConfigParser
from repeat import RepeatSetter

from datetime import datetime
from monai.losses import DiceLoss

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()

    if not args.shuffle:
        repeat_setter = RepeatSetter(seed=args.seed)
        repeat_setter()

    save_dir = os.path.abspath(os.path.join(args.save_dir, datetime.now().strftime('%Y%m%d%H%M')[2:]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_reader = DataReaderMSD(
        root_dir=args.root_dir,
        args=args
    )
    train_loader = data_reader.get_dataloader('train')
    valid_loader = data_reader.get_dataloader('valid')

    if args.model == 'UNet3D':
        model = UNet3D(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            n_channels=args.n_channels,
            norm_layer=args.norm_layer
        )
    elif args.model == 'VNet3D':
        model = VNet3D(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            n_channels=args.n_channels,
            layer_nums=[1, 2] + [3] * (len(args.n_channels) - 2)
        )
    elif args.model == 'UNetMONAI':
        norm_layer = {
            'BatchNorm': monai_layers.Norm.BATCH,
            'InstanceNorm': monai_layers.Norm.INSTANCE,
            'None': None
        }
        model = monai_nets.UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=args.n_channels,
            strides=[2] * len(args.n_channels),
            num_res_units=2,
            norm=norm_layer[args.norm_layer]
        )
    elif args.model == 'VNetMONAI':
        model = monai_nets.VNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            act="prelu"
        )
    else:
        raise ValueError()

    loss_fn = DiceLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True
    )

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr
        )
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.lr
        )
    else:
        raise ValueError('args.optimizer should be in ["Adam", "SGD"]')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.1,
        patience=5,
        threshold=1e-05,
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-08,
        eps=1e-08
    )

    early_stopping = EarlyStopping(
        model=model,
        save_dir=save_dir,
        patience=args.epochs // 10,
        min_delta=1e-03,
        stop_criterion='valid',
        save_interval=5,
        verbose=True
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=save_dir,
        device=device,
        valid_interval=1,
        args=args
    )

    config_parser = ConfigParser(
        save_dir=save_dir,
        device=device,
        args=args,
        data_reader=data_reader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        early_stopping=early_stopping,
    )

    trainer.run(args.epochs)
