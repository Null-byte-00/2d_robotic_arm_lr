from torch import nn
from torch.optim import SGD
import torch


class Model(nn.Module):
    def __init__(self,num_inputs=6, num_outputs=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(num_inputs, 20)
        self.lstm = nn.LSTM(20, 20)
        self.linear2 = nn.Linear(20, num_outputs)

        self.reset_state()

        self.loss = nn.MSELoss()
        self.optimizer = SGD(self.parameters(), lr=0.001)
    
    def reset_state(self):
        self.h_0 = torch.zeros(1,20)
        self.c_0 = torch.zeros(1,20)
    
    def forward(self, x):
        x = self.linear1(x)
        out, (self.h_0, self.c_0)= self.lstm(x,
                                             (self.h_0, self.c_0))
        return self.linear2(out)
    
    def train_sample(self, x, target):
        out = self.forward(x)
        loss = self.loss(out, target)
        loss.backward()
        self.optimizer.step()
        return loss
