import pickle
import numpy as np
import os
import csv
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

from cnn import build_CNN
from mfe import build_MFE

with open('../data/y.pickle', 'rb') as f:
    y = pickle.load(f).astype(np.int64)

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

args = parser.parse_args()

os.makedirs(f'../result/{args.seed}/{args.train_size}', exist_ok=True)

print(f'train size: {args.train_size}')

y_trainval, y_test = train_test_split(y, test_size=10000, random_state=args.seed, stratify=y)
y_trainval = y_trainval[:args.train_size]
y_train, y_val = train_test_split(y_trainval, test_size=0.2, random_state=args.seed, stratify=y_trainval)

cnn_log = build_CNN(args)
mfe_log = build_MFE(args)

with open(f'../result/{args.seed}/{args.train_size}/log.pickle', 'wb') as f:
    pickle.dump([cnn_log, mfe_log], f)

with open(f'../result/{args.seed}/{args.train_size}/y_hat_MFE.pickle', 'rb') as f:
    y_hat_MFE_trainval, y_hat_MFE_test = pickle.load(f)
with open(f'../result/{args.seed}/{args.train_size}/y_hat_CNN.pickle', 'rb') as f:
    y_hat_CNN_trainval, y_hat_CNN_test = pickle.load(f)

X_stack_trainval = np.concatenate([y_hat_CNN_trainval, y_hat_MFE_trainval], axis=1)
X_stack_test = np.concatenate([y_hat_CNN_test, y_hat_MFE_test], axis=1)
assert X_stack_trainval.shape[1] == 18

ridge = RidgeClassifier(alpha=args.alpha)
ridge.fit(X_stack_trainval, y_trainval)
y_hat_stack_test = ridge.predict(X_stack_test)

f1_macro_stack_test = f1_score(y_test, y_hat_stack_test, average='macro', labels=np.unique(y_train))
f1_micro_stack_test = f1_score(y_test, y_hat_stack_test, average='micro', labels=np.unique(y_train))

with open(f'../result/{args.seed}/{args.train_size}/ridge.pickle', 'wb') as f:
    pickle.dump(ridge, f)

# save test result as csv
is_file_exist = os.path.isfile('../result/result.csv')

with open(f'../result/result.csv', 'a') as f:
    writer = csv.writer(f)
    if not is_file_exist:
        writer.writerow(['args.seed', 'args.train_size', 'mode', 'f1_macro', 'f1_micro'])
    writer.writerow([args.seed, args.train_size, 'stack', f1_macro_stack_test, f1_micro_stack_test])