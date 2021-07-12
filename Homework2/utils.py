'''
Please add utililty functions here
'''
import torch
import matplotlib.pyplot as plt
from configs import *

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            model.to(device)

            predictions = model.predict(x)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

def plot_valid_train_accu(steps, val_accur, train_accur, title):
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(steps, val_accur, label='Validation')
    ax1.plot(steps, train_accur, label='Training')

    ax1.set_xlabel('Training steps')
    ax1.set_ylabel('Accuracy')

    ax1.legend()

    fig.suptitle(title)

    plt.draw()

    fig.savefig(PLOT_DIR + title +'.png')

