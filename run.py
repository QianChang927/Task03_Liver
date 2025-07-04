import torch
import os

from data import DataReader
from model import UNet3D
from train import Trainer
from repeat import enable_repeat
from monai.losses import DiceLoss

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
SHUFFLE = False

if __name__ == '__main__':
    if not SHUFFLE:
        enable_repeat()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.abspath(os.path.join('.', 'checkpoint', 'size_64_64_48_roi_64_64_32_sw_4_lr_1e-03_batch_4_no_square_no_resized'))

    data_reader = DataReader(root_dir='../Task_Dataset/Task03_Liver', shuffle=SHUFFLE,
                             train_dir='imagesTr', label_dir='labelsTr', test_dir='imagesTs',
                             remain_nums=None, val_scale=0.1, num_workers_loader=4)

    train_loader = data_reader.get_dataloader(target='train', batch_size=4)
    valid_loader = data_reader.get_dataloader(target='valid', batch_size=1)

    model = UNet3D(in_channels=1, out_channels=2, batch_norm=True)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75,
                                                           patience=2, threshold=1e-06, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-08, eps=1e-08)

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                      train_loader=train_loader, valid_loader=valid_loader, save_dir=save_dir,
                      device=device, valid_interval=1)
    trainer.run(300)

    torch.save(trainer.train_criteria, os.path.join(save_dir, 'train_criteria.pth'))
    torch.save(trainer.valid_criteria, os.path.join(save_dir, 'valid_criteria.pth'))