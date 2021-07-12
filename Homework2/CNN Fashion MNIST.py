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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def predict(self, data):
        self.eval()
        scores = self.forward(data)
        _, predicts = scores.max(1)
        return predicts

def train_cnn(batch_size, epoche, learning_rate, train_folds, val_index, best_validation_accuracy):
    title = "batch_szie: %d|Epoche: %d|Learning_rate: %f|K: %d" % (batch_size, epoche, learning_rate, K)
    warn("Start training, with\nbatch_szie: %d\nEpoche: %d\nLearning_rate: %f\nK: %d" % (batch_size, epoche, learning_rate, K))
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    info("Device: " + str(device))

    # Initial the CNN model
    model = CNN()
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

    info("Epoche %d finished, time usage:%f" % (e, time.time() - start_time))
    plot_valid_train_accu(step_nums, validation_accuracies, train_accuracies, title)

    model.eval()

    return best_validation_accuracy

train_set = get_trainset2d_norm()
folds = K_fold(train_set, 10)

best_validation_accuracy = 0
for learning_rate in [1, 0.1, 0.01, 0.001, 0.0001]:
    best_validation_accuracy = train_cnn(16, 10, learning_rate, folds, random.randint(0, 9), best_validation_accuracy)



    

# in_channels = 1
# num_classes = 10
# batch_size = 64
# num_epochs = 5

# train_dataset = get_trainset2d_norm()
# test_dataset = get_testset2d_norm()
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

#         data = data.to(device=device)
#         targets = targets.to(device=device)

#         scores = model(data)
#         loss = criterion(scores, targets)

#         optimizer.zero_grad()
#         loss.backward()

#         optimizer.step()


# def check_accuracy(loader, model, device):
#     num_correct = 0
#     num_samples = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#             model.to(device)

#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)

#     model.train()
#     return num_correct/num_samples

# print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
# print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")