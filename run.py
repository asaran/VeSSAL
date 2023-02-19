import numpy as np
import sys
import os
import argparse
from dataset import get_dataset, get_handler
import resnet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
from query_strategies import RandomSampling, BadgeSampling, \
                            StreamingSampling, LeastConfidence, \
                            CoreSet, StreamingRand

def make_choices_help(choices):
   return '[' + ', '.join(choices) + ']'

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--train_acc', help='training stopping criteria', type=float, default=0.99)
parser.add_argument('--zeta', help='z_t in equation 3', type=float, default=1)
parser.add_argument('--fill_random', default=True, action='store_true')
parser.add_argument('--single_pass', default=False, action='store_true')
parser.add_argument('--sort_by_pc', default=False, action='store_true')
parser.add_argument('--deterministic', default=False, action='store_true', help="in streaming random sampler, whether to select every kth samples deterministically or select randomly with fixed rate")
parser.add_argument('--embs', default="grad_embs", help="whether to use gradient embeddings (grad_embs) or penultimate layer embeddings (penultimate).", type=str)
parser.add_argument('--stream_sampler_early_stop', default=False, action='store_true')
parser.add_argument('--model', help='model - resnet or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset' , type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)
parser.add_argument('--rank', help='rank of the sample-wise fisher information matrix', type=int, default=1)
parser.add_argument('--cov_inv_scaling', help='covariance inverse scaling', type=float, default=100)
parser.add_argument('--activation', help=make_choices_help(['square','relu','sigmoid']), type=str, default='square', metavar='activation function for gradient descent autotuning')

opts = parser.parse_args()

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)
DATA_NAME = opts.data

# non-openml data defaults
args_pool = {'MNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'CIFAR10':
                {'n_epoch': 3, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
                 'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])},
                }

opts.nClasses = 10
args_pool['CIFAR10']['transform'] =  args_pool['CIFAR10']['transformTest'] # remove data augmentation
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']


if not os.path.exists(opts.path):
    os.makedirs(opts.path)


X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
opts.dim = np.shape(X_tr)[1:]
handler = get_handler(opts.data)

args = args_pool[DATA_NAME]
args['lr'] = opts.lr
args['train_acc'] = opts.train_acc

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, opts.nClasses)
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize


class pretrained(nn.Module):
     def __init__(self, nClasses=40):
        # from torchvision.models import resnet18, ResNet18_Weights
        super(pretrained, self).__init__()
        self.backbone = torch.load('pretrained_resnet.pt')
        self.backbone.fc = nn.Identity(512, 512)
        self.lin = nn.Linear(512, nClasses)
     def forward(self, x):
        emb = self.backbone(x)
        out = self.lin(emb)
        return out, emb

     def get_embedding_dim(self):
        return 512

# load specified network
if opts.model == 'mlp':
    net = mlpMod(opts.dim, embSize=opts.nEmb)
elif opts.model == 'pretrained':
    net =  pretrained(nClasses=opts.nClasses)
elif opts.model == 'resnet':
    net = resnet.ResNet18(num_classes=opts.nClasses)
else: 
    print('choose a valid model - mlp, resnet, or pretrained', flush=True)
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

args["zeta"] = float(opts.zeta)
args["fill_random"] = opts.fill_random
args["early_stop"] = opts.stream_sampler_early_stop
args["single_pass"] = opts.single_pass
args["sort_by_pc"] = opts.sort_by_pc
args['rank'] = opts.rank
args['cov_inv_scaling'] = opts.cov_inv_scaling
args['data'] = opts.data
args['activation'] = opts.activation
args['embs'] = opts.embs
args["deterministic"] = opts.deterministic


# Sort the data by first PC to simulate domain shift.
if args["sort_by_pc"]:
    pca = PCA(n_components=1)
    X_flat = X_tr.reshape(X_tr.shape[0], -1).astype(float)
    X_flat = StandardScaler().fit_transform(X_flat)
    reduced = pca.fit_transform(X_flat)
    inds = np.argsort(reduced[:, 0])
    X_tr = X_tr[inds]
    Y_tr = Y_tr[inds]


# set up the specified sampler
if opts.alg == 'rand': # random sampling (baseline)
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'conf': # confidence-based sampling (baseline)
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == "stream_rand": # uniform stream sampling (streaming baseline)
    strategy = StreamingRand(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'badge': # batch active learning by diverse gradient embeddings (SoTA)
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'coreset': # coreset sampling (basline)
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'vessal': # Our proposed method. 
    strategy = StreamingSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
else: 
    print('choose a valid acquisition function', flush=True)
    raise ValueError

get_pretrained = lambda : pretrained(nClasses=opts.nClasses)
args['pretrained_resnet'] = get_pretrained
args['net_type'] = opts.model

print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)

# round 0 accuracy
strategy.train()
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print(str(opts.nStart) + '\ttesting accuracy {}'.format(acc[0]), flush=True)

scan_per_round = (len(X_tr) - NUM_INIT_LB) // NUM_ROUND
strategy.allowed = np.zeros(strategy.n_pool, dtype=bool)
for rd in range(1, NUM_ROUND+1):
    if opts.single_pass:
        strategy.allowed[rd*scan_per_round: (rd+1)*scan_per_round] = True 
    print('Round {}'.format(rd), flush=True)

    # query
    output = strategy.query(NUM_QUERY)#, rd)
    q_idxs = output
    if len(q_idxs) == 0:
        break
    idxs_lb[q_idxs] = True

    # report weighted accuracy
    corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == Y_tr.numpy()[q_idxs]

    # update
    strategy.update(idxs_lb)
    strategy.train()

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]), flush=True)
    if sum(~strategy.idxs_lb) < opts.nQuery: 
        sys.exit('too few remaining points to query')
