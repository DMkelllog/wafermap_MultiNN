import pickle
import numpy as np
import torch
import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from torch.utils.data import DataLoader
from torchvision import transforms

from utils import softmax, CustomDataset, FNN, training, inference




batch_size = 512
lr = 1e-4
es_patience = 20

def build_MFE(args):

    with open('../data/X_MFE.pickle', 'rb') as f:
        X_MFE = pickle.load(f)
        X_MFE = X_MFE.astype(np.float32)

    with open('../data/y.pickle', 'rb') as f:
        y = pickle.load(f).astype(np.int64)

    X_MFE_trainval, X_MFE_test, y_trainval, y_test = train_test_split(X_MFE, y, test_size=10000, random_state=args.seed, stratify=y)
    X_MFE_trainval, y_trainval = X_MFE_trainval[:args.train_size], y_trainval[:args.train_size]
    X_MFE_train, X_MFE_val, y_train, y_val = train_test_split(X_MFE_trainval, y_trainval, test_size=0.2, random_state=args.seed, stratify=y_trainval)

    mode = 'MFE'
    print(f'{mode}')

    scaler = StandardScaler()
    X_MFE_train = scaler.fit_transform(X_MFE_train)
    X_MFE_val = scaler.transform(X_MFE_val)
    X_MFE_trainval = scaler.transform(X_MFE_trainval)
    X_MFE_test = scaler.transform(X_MFE_test)

    dataset_MFE_train = CustomDataset(torch.from_numpy(X_MFE_train), y_train)
    dataset_MFE_val = CustomDataset(torch.from_numpy(X_MFE_val), y_val)
    dataset_MFE_trainval = CustomDataset(torch.from_numpy(X_MFE_trainval), y_trainval)
    dataset_MFE_test = CustomDataset(torch.from_numpy(X_MFE_test), y_test)

    dataloader_MFE_train = DataLoader(dataset_MFE_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_MFE_val = DataLoader(dataset_MFE_val, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_MFE_trainval = DataLoader(dataset_MFE_trainval, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_MFE_test = DataLoader(dataset_MFE_test, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FNN(args).cuda()

    model, log = training(model, dataloader_MFE_train, dataloader_MFE_val, mode, args)

    f1_macro_MFE_test, f1_micro_MFE_test, y_hat_MFE_test = inference(model, dataloader_MFE_test, y_test, np.unique(y_train), args)
    f1_macro_trainval, f1_micro_trainval, y_hat_MFE_trainval = inference(model, dataloader_MFE_trainval, y_trainval, np.unique(y_train), args)

    y_hat_MFE_trainval, y_hat_MFE_test = softmax(y_hat_MFE_trainval), softmax(y_hat_MFE_test)
    # save trainval result
    with open(f'../result/{args.seed}/{args.train_size}/y_hat_{mode}.pickle', 'wb') as f:
        pickle.dump([y_hat_MFE_trainval, y_hat_MFE_test], f)

    is_file_exist = os.path.isfile('../result/result.csv')
    with open(f'../result/result.csv', 'a') as f:
        writer = csv.writer(f)
        if not is_file_exist:
            writer.writerow(['args.seed', 'args.train_size', 'mode', 'f1_macro', 'f1_micro'])
        writer.writerow([args.seed, args.train_size, 'MFE', f1_macro_MFE_test, f1_micro_MFE_test])
    return log

