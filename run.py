import os
import json
import torch
import argparse

from datetime import datetime
from data import DataReader
from model import UNet3D
from train import Trainer
from repeat import enable_repeat
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_tuple(value: str) -> tuple[int]:
    try:
        value = value.strip()
        if value.startswith('(') and value.endswith(')'):
            value = value[1:-1]
        value = value.split(',')
        value = [s.strip() for s in value]
        return tuple(map(int, value))
    except:
        raise argparse.ArgumentTypeError(f'Invalid tuple forms: {value}. Use "(1, 2, 3)" or "1, 2, 3"')

def add_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=500, help='训练轮次')
    parser.add_argument('-b', '--batch', type=int, default=4, help='训练集Dataloader的batch_size')
    parser.add_argument('-l', '--lr', type=int, default=1e-03, help='优化器学习率')
    parser.add_argument('-s', '--shuffle', action="store_true", help='是否启用随机化')
    parser.add_argument('-i', '--input', type=str, help='数据集所在位置')
    parser.add_argument('-o', '--output', type=str, default='./checkpoint', help='模型保存位置')
    parser.add_argument('-q', '--squared_pred', action="store_true", help='DiceLoss参数squared_pred是否置为True')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='训练集Dataloader的num_workers')
    parser.add_argument('-r', '--remains', type=int, default=None, help='数据集保留个数')
    parser.add_argument('-v', '--val_scale', type=float, default=0.1, help='验证集占训练·验证集比例')
    parser.add_argument('-S', '--size', type=parse_tuple, default="(64, 64, 64)", help='数据预处理Transforms后输出的向量大小')
    parser.add_argument('-R', '--roi', type=parse_tuple, default="(64, 64, 64)", help='滑动窗口大小')
    parser.add_argument('-W', '--sw_batch', type=int, default=4, help='滑动窗口batch_size')
    parser.add_argument('-m', '--model', type=str, choices=['UNet3D', 'UNetMONAI'], default='UNet3D', help='训练所选模型')
    parser.add_argument('-L', '--layer', type=str, choices=['BatchNorm', 'InstanceNorm', 'None'], default='BatchNorm', help='Relu激活层前的添加层')
    return parser

def add_config_json(args: argparse.Namespace, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, 'config.json')

    config_dict = vars(args)
    print(json.dumps(config_dict, indent=4))

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

if __name__ == '__main__':
    parser = add_arg_parser()
    args = parser.parse_args()

    if not args.shuffle:
        enable_repeat()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.abspath(os.path.join(args.output, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    add_config_json(args, save_dir)

    data_reader = DataReader(root_dir=args.input, train_dir='imagesTr', label_dir='labelsTr',
                             test_dir='imagesTs', args=args)

    train_loader = data_reader.get_dataloader(target='train', batch_size=args.batch)
    valid_loader = data_reader.get_dataloader(target='valid', batch_size=1)

    if args.model == 'UNet3D':
        model = UNet3D(in_channels=1, out_channels=2, norm_layer=args.layer)
    elif args.model == 'UNetMONAI':
        norm_layer = {
            'BatchNorm': Norm.BATCH,
            'InstanceNorm': Norm.INSTANCE,
            'None': None
        }

        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=norm_layer[args.layer]
        )
    else:
        raise ValueError('args.model should be in ["UNet3D", "UNetMONAI"]')

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=args.squared_pred)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                           patience=2, threshold=1e-06, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-08, eps=1e-08)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                      train_loader=train_loader, valid_loader=valid_loader, save_dir=save_dir,
                      device=device, valid_interval=1, args=args)

    trainer.run(args.epochs)
    torch.save(trainer.train_criteria, os.path.join(save_dir, 'train_criteria.pth'))
    torch.save(trainer.valid_criteria, os.path.join(save_dir, 'valid_criteria.pth'))