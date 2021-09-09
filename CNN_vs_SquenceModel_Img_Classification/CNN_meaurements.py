from os import stat
from numpy.core.fromnumeric import reshape
import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.transforms import ToTensor
from configs import *
from data_manager import get_testset2d_norm
from data_manager import get_trainset2d_norm
from data_manager import get_loader
from data_manager import K_fold
from data_manager import statistics
from torch.nn import RNN
from CNN_stride import CNN_Stride
from utils import plot_confusion_heatmap
from utils import check_accuracy
import seaborn as sns
from CNN import CNN
from CNN_no_pooling import CNN_Small_Conv
from utils import cal_matrics


test = get_testset2d_norm()
train = get_trainset2d_norm()

cnn = CNN()
cnn_stride = CNN_Stride()
cnn_small = CNN_Small_Conv()

cnn.load_state_dict(torch.load("/home/tai/XDF_Project/MIT-Reserach/Homework2/trained_models/batch_szie: 16|Epoche: 10|Learning_rate: 0.001000|K: 10|Accuracy: 0.9120000004768372"))
cnn_stride.load_state_dict(torch.load("/home/tai/XDF_Project/MIT-Reserach/Homework2/trained_models/STRIDE|batch_szie: 16|Epoche: 20|Learning_rate: 0.000100|K: 10|Accuracy: 0.9138333201408386"))
cnn_small.load_state_dict(torch.load("/home/tai/XDF_Project/MIT-Reserach/Homework2/trained_models/NO_POOL|batch_szie: 16|Epoche: 20|Learning_rate: 0.001000|K: 10|Accuracy: 0.906166672706604"))
cnn_stride.eval()


plot_confusion_heatmap(cnn_stride, test)

print(check_accuracy(get_loader(test, 32), cnn, torch.device('cpu')))
print(check_accuracy(get_loader(test, 32), cnn_stride, torch.device('cpu')))
print(check_accuracy(get_loader(test, 32), cnn_small, torch.device('cpu')))

print(cal_matrics(cnn, test))
print(cal_matrics(cnn_stride, test))
print(cal_matrics(cnn_small, test))

print(check_accuracy(get_loader(train, 32), cnn, torch.device('cpu')))
print(check_accuracy(get_loader(train, 32), cnn_stride, torch.device('cpu')))
print(check_accuracy(get_loader(train, 32), cnn_small, torch.device('cpu')))