import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from lj import *
import math
import random

ground_truth_pos = np.array([[0.3657214990,1.5499815375,0.0190248507],
         [-0.3544443293,-1.5454469349,-0.1500547478],
          [0.0224253734,-0.5199411287,-1.8824579377],
          [0.2942511696,0.6484680230,-1.8186317939],
         [-0.9487600199,0.7579319084,1.3367980034],
         [ 0.9811424830,0.3007659257,1.4861170249],
          [0.7125725663,-0.8537530735,1.4230253025],
         [-1.2173709678,-0.3965809882,1.2737204085],
          [1.4907145113,-0.3835944414,0.6725102450],
         [-1.5854581538,0.3451379985,0.4345190102],
          [1.0293483159,-0.1612730069,-1.4314479661],
         [-0.7724396134,0.2655318801,-1.5707984280],
          [0.8500938283,0.8097688296,-0.8107984796],
          [0.4260809805,-1.0126532821,-0.9103558710],
         [-0.6953693576,-0.7470080944,-0.9970933215],
         [-0.2713590232,1.0754365712,-0.8975251087],
          [1.1032022459,0.6869507065,0.3791446186],
         [-1.1474532709,-0.7049307969,0.1381832970],
         [-0.7229684543,1.1195622973,0.2378511990],
          [0.6787076748,-1.1375211976,0.2794507197],
         [-0.1357528338,-0.0550771347,1.5861418890],
          [0.1326138545,0.8762804378,0.9690598049],
         [-0.2899829399,-0.9401255908,0.8698279129],
          [0.9360095623,-0.2009685418,-0.3076161678],
         [-0.8713780839,0.2271807215,-0.4474279742],
          [0.0837812503,0.0339965666,-0.9790099656],
          [0.4580976327,-0.1437364886,0.6800785028],
         [-0.5677038315,0.0992666994,0.6007153967],
         [-0.1143275044,-0.5218957052,-0.1201619299],
          [0.1300054359,0.5282463026,-0.0627884942]])


data = np.load('data.npz')
finalposition = data["basinpos"]
energy = data['energy']

class QNet(nn.Module):

    def __init__(self, ntoken: int, d_model: int, 
                  dropout: float = 0.5):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(ntoken*3, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
        )
        self.ntoken = ntoken
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        for l in self.layers:
            try:
                l.weight.data.uniform_(-initrange, initrange)
            except:
                pass

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src=src.view(-1, self.ntoken * 3)
        output = self.layers(src)
        return output

#qnet = QNet(ntoken=30, d_model=512, nhead=8, d_hid=2048, nlayers=6).to("cuda")
qnet = QNet(ntoken=30, d_model=128).to("cuda")
randdata = torch.normal(0, 1, (128, 30, 3)).to("cuda")
target = torch.normal(0, 1, (128,)).to("cuda")
optimizer = torch.optim.Adam(qnet.parameters(), lr=0.0001)

numdata = finalposition.shape[0]
batch_size = 128
Loss = nn.MSELoss()
rounds = 1000
for _ in range(rounds):
    index = random.sample(list(range(numdata)), 128)
    feature = torch.FloatTensor(finalposition[index, :, :]).to("cuda")
    print(feature.shape)
    target = torch.FloatTensor(energy[index]).to("cuda")
    target /= 100.
    target = torch.clamp(target, max=0)
    predict = qnet(feature).view(-1)
    loss = Loss(predict, target)
    qnet.zero_grad()
    loss.backward()
    optimizer.step()
    print(_, loss)
    if _ == rounds - 1:
        print(target[:10] - predict[:10], target[:10])
