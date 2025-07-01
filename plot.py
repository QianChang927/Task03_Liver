import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    train_loss = torch.load('./checkpoint/train_criteria.pth')
    valid_dice = torch.load('./checkpoint/valid_criteria.pth')
    plt.figure('train', (12, 6))

    x = [i + 1 for i in range(len(train_loss))]
    train_loss = [-loss for loss in train_loss]

    plt.subplot(1, 2, 1)
    plt.title('Epoch Average Loss')
    plt.xlabel('Epoch')
    plt.plot(x, train_loss)

    plt.subplot(1, 2, 2)
    plt.title('Valid Mean Dice')
    plt.xlabel('Epoch')
    plt.plot(x, valid_dice)

    plt.show()

    print(f"min train loss: {min(train_loss):.4f} at epoch {train_loss.index(min(train_loss)) + 1}")
    print(f"max valid dice: {max(valid_dice):.4f} at epoch {valid_dice.index(max(valid_dice)) + 1}")