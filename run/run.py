import pickle
import numpy as np
import torch
import os
import csv
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
from torchvision import transforms

from utils import CustomDataset, MultiNN, training, inference

def build_MultiNN(args):
    with open(f'../data/X_CNN.pickle', 'rb') as f:
        X_CNN = pickle.load(f)
        X_CNN = np.expand_dims(X_CNN, 3).astype(np.float32) / 255
        X_CNN = X_CNN.transpose(0, 3, 1, 2)

    with open(f'../data/X_MFE.pickle', 'rb') as f:
        X_MFE = pickle.load(f)

    with open('../data/y.pickle', 'rb') as f:
        y = pickle.load(f).astype(np.int64)

    X_CNN_trainval, X_CNN_test, X_MFE_trainval, X_MFE_test, y_trainval, y_test = train_test_split(X_CNN, X_MFE, y, test_size=10000, random_state=args.seed, stratify=y)
    X_CNN_trainval, X_MFE_trainval, y_trainval = X_CNN_trainval[:args.train_size], X_MFE_trainval[:args.train_size], y_trainval[:args.train_size]
    X_CNN_train, X_CNN_val, X_MFE_train, X_MFE_val, y_train, y_val = train_test_split(X_CNN_trainval, X_MFE_trainval, y_trainval, test_size=0.2, random_state=args.seed)

    mode = 'MultiNN'
    print(f'{args.seed} {args.train_size}')

    augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip()])

    scaler = StandardScaler()
    X_MFE_train = scaler.fit_transform(X_MFE_train)
    X_MFE_val = scaler.transform(X_MFE_val)
    X_MFE_test = scaler.transform(X_MFE_test)

    dataset_train = CustomDataset(torch.from_numpy(X_CNN_train), torch.from_numpy(X_MFE_train), y_train, transform=augmentations)
    dataset_val = CustomDataset(torch.from_numpy(X_CNN_val), torch.from_numpy(X_MFE_val), y_val)
    dataset_test = CustomDataset(torch.from_numpy(X_CNN_test), torch.from_numpy(X_MFE_test), y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiNN(args, pretrained=True).cuda()

    model, log = training(model, dataloader_train, dataloader_val, mode, args)

    f1_macro_test, f1_micro_test, y_hat_test = inference(model, dataloader_test, y_test, np.unique(y_train), args)

    with open(f'../result/{args.seed}/{args.train_size}/y_hat_{mode}.pickle', 'wb') as f:
        pickle.dump([y_hat_test], f)

    is_file_exist = os.path.isfile('../result/result.csv')
    with open(f'../result/result.csv', 'a') as f:
        writer = csv.writer(f)
        if not is_file_exist: 
            writer.writerow(['args.seed', 'args.train_size', 'mode', 'f1_macro', 'f1_micro'])
        writer.writerow([args.seed, args.train_size, mode, f1_macro_test, f1_micro_test])
    return log

if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--train_size', type=int, default=500, help='train_size')
    parser.add_argument('--alpha', type=float, default=0.1, help='ridge regularization parameter')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--max_epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--print_freq', type=int, default=0, help='training print frequency')
    parser.add_argument('--vgg', type=int, default=16, help='vgg layer')
    parser.add_argument('--bn', type=int, default=0, help='batch normalization')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--es_patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()

    os.makedirs(f'../result/{args.seed}/{args.train_size}', exist_ok=True)

    log = build_MultiNN(args)