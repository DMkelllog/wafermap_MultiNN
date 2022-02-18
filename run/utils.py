from cProfile import label
from torch.utils.data import Dataset
from torchvision.models import vgg13, vgg16, vgg19, vgg13_bn, vgg16_bn, vgg19_bn
import torch
from torch import nn
from torch import optim
import time
import pickle
import numpy as np
from sklearn.metrics import f1_score

def softmax(x):
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x


class CustomDataset(Dataset):
    
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


class CNN(nn.Module):
    def __init__(self, args, pretrained=True):
        super(CNN, self).__init__()
        self.f = []
        if args.bn:
            if args.vgg==13: architecture= vgg13_bn(pretrained=pretrained)
            elif args.vgg==16: architecture= vgg16_bn(pretrained=pretrained)
            elif args.vgg==19: architecture= vgg19_bn(pretrained=pretrained)
        else:
            if args.vgg==13: architecture= vgg13(pretrained=pretrained)
            elif args.vgg==16: architecture= vgg16(pretrained=pretrained)
            elif args.vgg==19: architecture= vgg19(pretrained=pretrained)
        
        architecture.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        architecture.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        architecture.classifier = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(512, 9))
        self.cnn = architecture

    def forward(self, x):
        return self.cnn(x)


class FNN(nn.Module):
    def __init__(self, args):
        super(FNN, self).__init__()
        if args.activation=='relu': self.activation = nn.ReLU()
        elif args.activation=='tanh': self.activation = nn.Tanh()
        self.fnn = nn.Sequential(
            nn.Linear(59, args.hidden_size), self.activation, nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size), self.activation, nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, 9))

    def forward(self, x):
        return self.fnn(x)


def training(model, dataloader_train, dataloader_val, mode, args):    
    result_path = f'../result/{args.seed}/{args.train_size}'

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7, verbose=False)

    train_log, val_log = np.zeros(args.max_epochs), np.zeros(args.max_epochs)

    for epoch in range(args.max_epochs):
        start = time.time()

        model.train()
        total_loss, total_num = 0, 0
        for x, y in dataloader_train:
            x, y = x.cuda(), y.cuda()
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += y.size(0)
        train_loss = total_loss / total_num

        model.eval()
        total_loss, total_num = 0, 0
        with torch.no_grad():
            for x, y in dataloader_val:
                x, y = x.cuda(), y.cuda()

                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                total_loss += loss.item()
                total_num += y.size(0)
            val_loss = total_loss / total_num
        if args.print_freq == 0: pass
        elif epoch % args.print_freq == 0:
            print(f'Epoch: {epoch:3d} Train loss: {train_loss:0.6f} Val loss: {val_loss:0.6f} {(time.time() - start)/60*args.print_freq:0.1f}min')

        lr_scheduler.step(val_loss)

        train_log[epoch] = train_loss
        val_log[epoch] = val_loss

        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(model.state_dict(), f'{result_path}/{mode}.pt')
            best_val_epoch = epoch

        elif np.argmin(val_log[:epoch + 1]) <= epoch - args.es_patience:
            break
        
    log = {'train_loss': train_log, 'val_loss': val_log}
    if args.print_freq == 0: pass
    else: print(f'training terminated at epoch {epoch}. returning best_val_epoch: {best_val_epoch}')

    with open(f'{result_path}/log_{mode}.pickle', 'wb') as f:
        pickle.dump(log, f)
    model.load_state_dict(torch.load(f'{result_path}/{mode}.pt'))
    return model, log


def inference(model, dataloader_test, y_test, y_set, args):
    model.eval()
    y_hat = []
    with torch.no_grad():
        for x, y in dataloader_test:
            x, y = x.cuda(), y.cuda()
            y_pred = model(x)
            y_hat.append(y_pred.cpu().numpy())
    y_hat = np.vstack(y_hat)
    f1_macro = f1_score(y_test, y_hat.argmax(1), average='macro', labels=y_set)
    f1_micro = f1_score(y_test, y_hat.argmax(1), average='micro', labels=y_set)
    if args.print_freq == 0: pass
    else: print(f'f1 score: {f1_macro:0.4f} {f1_micro:0.4f}')
    return f1_macro, f1_micro, y_hat