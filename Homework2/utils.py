'''
Please add utililty functions here
'''
from numpy.lib.function_base import average
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from configs import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from data_manager import get_loader
from torch.utils.data import DataLoader

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
    fig, ax1 = plt.subplots(dpi=600)
    ax1.plot(steps, val_accur, label='Validation')
    ax1.plot(steps, train_accur, label='Training')

    ax1.set_xlabel('Training steps')
    ax1.set_ylabel('Accuracy')

    ax1.legend()

    fig.suptitle(title)

    plt.draw()

    fig.savefig(PLOT_DIR + title +'.png')

def plot_confusion_heatmap(model, dataset):
    loader = DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=True,
        num_workers=4
    )
    loader = get_loader(dataset, len(dataset))
    model.eval()
    model.to('cpu')
    _, (data, labels) = enumerate(loader).__next__()
    predicts = model.predict(data)

    conf_mat = confusion_matrix(labels, predicts)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=600)
    ax = sns.heatmap(
        conf_mat, 
        xticklabels=dataset.classes, 
        yticklabels=dataset.classes, 
        annot=True, 
        cmap="YlGnBu", 
        fmt='d', 
        ax=ax, 
        linewidths=.8
    )

    plt.draw()

    fig.savefig(PLOT_DIR + 'heat_map.png')

    return conf_mat
    
def cal_matrics(model, dataset):
    loader = get_loader(dataset, len(dataset))
    model.eval()
    model.to('cpu')
    _, (data, labels) = enumerate(loader).__next__()
    predicts = model.predict(data)

    f1 = f1_score(labels, predicts, average='macro')
    precision = precision_score(labels, predicts, average='macro')
    recall = recall_score(labels, predicts, average='macro')

    return f1, precision, recall