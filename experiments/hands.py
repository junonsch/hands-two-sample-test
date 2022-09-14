from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler
import os
import numpy as np
import torch
data_dir = "../../transfer_hands/data/hands/Hands/Hands"


DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
       transforms.RandomPerspective(distortion_scale=0.4),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          DATA_TRANSFORMS[x])
                  for x in ['train', 'val', 'test']}


def weighted_sampling(data):
    labels = [y[1] for y in data.imgs]
    labels_unique, counts = np.unique(labels, return_counts = True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[e] for e in labels]
    sampler = WeightedRandomSampler(example_weights, len(labels))
    
    return sampler

train_sampler = weighted_sampling(image_datasets["train"])
val_sampler = weighted_sampling(image_datasets["val"])
test_sampler = weighted_sampling(image_datasets["test"])

samplers = {"train": train_sampler,
           "val": None,#val_sampler,
           "test": None} #test_sampler}

train_reg = [x for x,y in image_datasets["train"].samples if y==1]
train_irreg = [x for x,y in image_datasets["train"].samples if y==0]

dataloaders = {"reg": torch.utils.data.DataLoader(train_reg, batch_size=32,
                                             sampler=train_sampler, num_workers=4),
                "irreg": torch.utils.data.DataLoader(train_irreg, batch_size=32,
                                             sampler=train_sampler, num_workers=4)
              }
