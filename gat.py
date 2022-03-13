import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import *

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#testing the representation power of GAT

#TODO: a function: locate to graph
#TODO: init NN with (N, config)


parser = argparse.ArgumentParser(description='GAT QNet')
parser.add_argument('--nfeat', type=int, default=128,
                    help='the number of features for each node')
parser.add_argument('--nhid', type=int, default=128,
                    help='dimension of hidden layers')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--gatoutdim', type=int, default=1, help='the dimension of the GAT output.')
parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--natoms', type=int, default=8, help='Number of atoms in LJ system.')

args = parser.parse_args()
#testing Attention layer
att = GraphAttentionLayer(10, 10, args.dropout, args.alpha).to("cuda")

N, minsep, maxV = 10, 1.5, 3
def loc2graph(loc, radius):
    #loc: np.array with shape (N, 3)
    N = loc.shape[0]
    dis = 100 * np.ones((N, N))
    for i in range(N):
        diff = loc - loc[i, :]
        dis[i, :] = np.linalg.norm(diff, axis = 1)
        dis[i, i] = 100
    adj = np.where(dis<radius, 1, 0)
    return adj

init = generate_random_data(N, minsep, maxV)
loc, suc = init[0], init[1]
adj = torch.tensor(loc2graph(loc, 3)).to("cuda")
print(loc)
print(adj)

init_embedding = nn.Linear(3, 10).to("cuda")
loc = torch.FloatTensor(loc).to("cuda")
emb = init_embedding(loc)
gatemb = att(emb, adj)

print(gatemb.shape)

class GAT(nn.Module):
    def __init__(self, config):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = config.dropout
        self.init_embedding = nn.Linear(3, config.nfeat)
        self.attentions = [GraphAttentionLayer(config.nfeat, config.nhid, dropout=config.dropout, alpha=config.alpha, concat=True) for _ in range(config.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(config.nhid * config.nheads, config.gatoutdim, dropout=config.dropout, alpha=config.alpha, concat=False)

    def forward(self, x, adj):
        x = self.init_embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

gat = GAT(args).to("cuda")
final = gat(loc, adj)
print(final.shape)

data = np.load('data.npz')
finalposition = data["basinpos"]
energy = data['energy']
optimizer = torch.optim.Adam(gat.parameters(), lr=0.0001)
numdata = finalposition.shape[0]
batch_size = 2
Loss = nn.MSELoss()
rounds = 1

def batch2adj(data, batchidx):
    total_nodes = 0
    for idx in batchidx:
        total_nodes += data[idx].shape[0]
    ret = np.zeros((total_nodes, total_nodes))
    cur = 0
    for idx in batchidx:
        adj = loc2graph(data[idx])
        nodes = adj.shape[0]
        ret[cur:cur+nodes, cur:cur+nodes] = adj
        cur += nodes 
    return ret

for _ in range(rounds):
    index = random.sample(list(range(numdata)), batch_size)
    adj = batch2adj(finalposition, index)
    

    # feature = torch.FloatTensor(finalposition[index, :, :]).to("cuda")
    # print(feature.shape)
    # target = torch.FloatTensor(energy[index]).to("cuda")
    # target /= 100.
    # target = torch.clamp(target, max=0)
    # predict = qnet(feature).view(-1)
    # loss = Loss(predict, target)
    # qnet.zero_grad()
    # loss.backward()
    # optimizer.step()
    # print(_, loss)
    # if _ == rounds - 1:
    #     print(target[:10] - predict[:10], target[:10])

