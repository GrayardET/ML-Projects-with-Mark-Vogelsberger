import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from configs import MEAN
from configs import STD
from configs import FASHION_ROOT

'''
Get the normalised 2d FashionMNIST training set object, 
data points are format in:
(label, img)
label is an int in [0 - 9]
img is a (1, 28, 28) float tensor
'''
def get_trainset2d_norm():
    train_set = torchvision.datasets.FashionMNIST(
        root=FASHION_ROOT,
        train=True, # Read local data if has downloaded
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))
        ])
    )

    return train_set

'''
Get the normalised 2d FashionMNIST test set object, 
data points are format in:
(label, img)
img is a (1, 28, 28) tensor
'''
def get_testset2d_norm():
    test_set = torchvision.datasets.FashionMNIST(
        root=FASHION_ROOT,
        train=False,
        download=True,  # Read local data if has downloaded
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    )

    return test_set

'''
Randomly split a dataset object into K subdatasets,
used for K cross validation
return a list of K subdatasets
'''
def K_fold(dataset, K):
    splits = random_split(
        dataset,
        [len(dataset)//K for i in range(K)]
    )

    return splits

'''
Convert a dataset/subdataset into a loader with random order,
parallel process the data, fastest way to load data.
How to use DataLoader?

for img_label_batch in enumerate(loader):
    do someting

'''
def get_loader(dataset, batch_size=4):
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return loader

'''
Calculate the pixel-wise statistics of all training set images
'''
def statistics():
    train_set = torchvision.datasets.FashionMNIST(
        root=FASHION_ROOT,
        train=True,
        download=True,  # Read local data if has downloaded
        transform=transforms.ToTensor()
    )

    # train_set.data are uint8 data in range [0, 255]
    data = train_set.data.float() / 255
    flat = torch.flatten(data)

    mean = flat.mean()
    std = flat.std()

    print("MEAN: \t%f\nSTD: \t%f" % (mean, std))

    return mean, std
