import os
import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    root_dir = 'checkpoint'
    expr_dir = 'size_48_roi_48_sw_4_lr_1e-03_batch_4_with_square'

    train_dict = torch.load(os.path.join(root_dir, expr_dir, 'train_criteria.pth'))
    valid_dict = torch.load(os.path.join(root_dir, expr_dir, 'valid_criteria.pth'))
    plt.figure('train', (12, 6))

    train_loss = [loss for loss in train_dict['loss']]
    train_dice = [dice for dice in train_dict['dice']]
    valid_dice = [dice for dice in valid_dict['dice']]
    x = [i + 1 for i in range(len(train_loss))]

    plt.subplot(1, 3, 1)
    plt.title('Epoch Average Loss')
    plt.xlabel('Epoch')
    plt.plot(x, train_loss)

    plt.subplot(1, 3, 2)
    plt.title('Epoch Average Dice')
    plt.xlabel('Epoch')
    plt.plot(x, train_dice)

    plt.subplot(1, 3, 3)
    plt.title('Valid Mean Dice')
    plt.xlabel('Epoch')
    plt.plot(x, valid_dice)

    plt.show()

    print(f"min train loss: {min(train_loss):.4f} at epoch {train_loss.index(min(train_loss)) + 1}")
    print(f"max train dice: {max(train_dice):.4f} at epoch {train_dice.index(max(train_dice)) + 1}")
    print(f"max valid dice: {max(valid_dice):.4f} at epoch {valid_dice.index(max(valid_dice)) + 1}")