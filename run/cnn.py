import pickle
import numpy as np
import torch
import os
import csv

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms

from utils import CustomDataset, CNN, training, inference




batch_size = 512
lr = 1e-4
es_patience = 20

def build_CNN(args):

    with open('../data/X_CNN_64.pickle', 'rb') as f:
        X_CNN = pickle.load(f)
        X_CNN = X_CNN.astype(np.float32)
        X_CNN = X_CNN.transpose(0, 3, 1, 2)

    with open('../data/y.pickle', 'rb') as f:
        y = pickle.load(f).astype(np.int64)

    X_CNN_trainval, X_CNN_test, y_trainval, y_test = train_test_split(X_CNN, y, test_size=10000, random_state=args.seed, stratify=y)
    X_CNN_trainval, y_trainval = X_CNN_trainval[:args.train_size], y_trainval[:args.train_size]
    X_CNN_train, X_CNN_val, y_train, y_val = train_test_split(X_CNN_trainval, y_trainval, test_size=0.2, random_state=args.seed, stratify=y_trainval)

    mode = 'CNN'
    print(f'{mode}')

    augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip()])

    dataset_CNN_train = CustomDataset(torch.from_numpy(X_CNN_train), y_train, transform=augmentations)
    dataset_CNN_val = CustomDataset(torch.from_numpy(X_CNN_val), y_val)
    dataset_CNN_trainval = CustomDataset(torch.from_numpy(X_CNN_trainval), y_trainval)
    dataset_CNN_test = CustomDataset(torch.from_numpy(X_CNN_test), y_test)

    dataloader_CNN_train = DataLoader(dataset_CNN_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_CNN_val = DataLoader(dataset_CNN_val, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_CNN_trainval = DataLoader(dataset_CNN_trainval, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_CNN_test = DataLoader(dataset_CNN_test, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CNN(args, pretrained=True).cuda()

    model, log = training(model, dataloader_CNN_train, dataloader_CNN_val, mode, args)

    f1_macro_CNN_test, f1_micro_CNN_test, y_hat_CNN_test = inference(model, dataloader_CNN_test, y_test, np.unique(y_train), args)
    f1_macro_trainval, f1_micro_trainval, y_hat_CNN_trainval = inference(model, dataloader_CNN_trainval, y_trainval, np.unique(y_train), args)

    with open(f'../result/{args.seed}/{args.train_size}/y_hat_{mode}.pickle', 'wb') as f:
        pickle.dump([y_hat_CNN_trainval, y_hat_CNN_test], f)

    is_file_exist = os.path.isfile('../result/result.csv')
    with open(f'../result/result.csv', 'a') as f:
        writer = csv.writer(f)
        if not is_file_exist:
            writer.writerow(['args.seed', 'args.train_size', 'mode', 'f1_macro', 'f1_micro'])
        writer.writerow([args.seed, args.train_size, mode, f1_macro_CNN_test, f1_micro_CNN_test])
    return log

