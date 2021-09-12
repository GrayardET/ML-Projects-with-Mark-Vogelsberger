# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_manager import get_trainset2d_norm
from data_manager import get_testset2d_norm
from data_manager import get_loader
from data_manager import K_fold
from configs import *
import time
import logging
from logging import info
from logging import warn
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
from utils import check_accuracy
from utils import plot_valid_train_accu
import random

class CNN_Stride(nn.Module):
    def __init__(self):
        super(CNN_Stride, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
        )        
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def predict(self, data):
        self.eval()
        scores = self.forward(data)
        _, predicts = scores.max(1)
        return predicts

def train_cnn(batch_size, epoche, learning_rate, train_folds, val_index, best_validation_accuracy):
    title = "STRIDE|batch_szie: %d|Epoche: %d|Learning_rate: %f|K: %d" % (batch_size, epoche, learning_rate, K)
    warn("Start training, with\nbatch_szie: %d\nEpoche: %d\nLearning_rate: %f\nK: %d" % (batch_size, epoche, learning_rate, K))
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    info("Device: " + str(device))

    # Initial the CNN model
    model = CNN_Stride()
    model.to(device)
    model.train()   # Switch model to train mode
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initial some recordings
    step_nums = []
    train_accuracies = []
    validation_accuracies = []
    step_num = 1

    # Saperate the training and validation set
    validation_set = train_folds[val_index]
    train_sets = []
    for i in range(len(train_folds)):
        if i != val_index:
            train_sets.append(train_folds[i])
    
    # Training epoche by epoche
    for e in range(epoche):
        torch.cuda.empty_cache()
        info("Start epoche %d:" % (e))
        start_time = time.time()

        for train_set in tqdm(train_sets):
            train_loader = get_loader(train_set, batch_size)
            valid_loader = get_loader(validation_set, batch_size)

            for _, (batch, labels) in enumerate(train_loader):
                batch = batch.to(device)
                labels = labels.to(device)

                scores = model(batch)
                loss = criterion(scores, labels)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                # Finished one more step
                step_num += 1
                
            # Calculate the accuracies on validation set and training set
            train_accur = check_accuracy(train_loader, model, device)
            valid_accur = check_accuracy(valid_loader, model, device)
            step_nums.append(step_num)
            train_accuracies.append(train_accur)
            validation_accuracies.append(valid_accur)
            info("Accuracy at step %d:\ntrain: %f\tvalid: %f" % (step_num, train_accur, valid_accur))

            if valid_accur > best_validation_accuracy and epoche > 1:
                torch.save(model.state_dict(), MODEL_DIR + title + "|Accuracy: " + str(float(best_validation_accuracy)))
                best_validation_accuracy = valid_accur

    info("Epoche %d finished, time usage:%f" % (e, time.time() - start_time))
    plot_valid_train_accu(step_nums, validation_accuracies, train_accuracies, title)

    model.eval()

    return best_validation_accuracy

if __name__ == "__main__":
    train_set = get_trainset2d_norm()
    folds = K_fold(train_set, 10)

    best_validation_accuracy = 0.87
    for learning_rate in [0.001, 0.0001, 0.1, 0.01]:
        for i in range(3):
            best_validation_accuracy = train_cnn(
                batch_size=16, 
                epoche=20, 
                learning_rate=learning_rate, 
                train_folds=folds, 
                val_index=random.randint(0, 9), 
                best_validation_accuracy=best_validation_accuracy
            )