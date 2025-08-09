import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as func
import math


class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(TransitionModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.sigmoid = nn.Sigmoid()
        self.act = nn.Tanh()
        
        # inital hidden state
        self.hidden_state = torch.rand(self.batch_size, self.hidden_dim).to(self.device)

        # build neural network
        self.W_z = nn.Linear(self.input_dim, self.hidden_dim)

        self.U_z = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.W_r = nn.Linear(self.input_dim, self.hidden_dim)

        self.U_r = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.U = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.W_c = nn.Linear(self.input_dim, self.input_dim)

        self.U_c = nn.Linear(self.hidden_dim, self.input_dim)

        self.h2s = nn.Linear(self.hidden_dim, self.input_dim)

        self.initialize_parameters()


    def initialize_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                
                init.xavier_normal_(param)
                # init.xavier_uniform_(param)


    def forward(self, inputs):

        s_left = torch.squeeze(inputs[0], 2)
        s_left = func.normalize(s_left, p=2, dim=1, eps=1e-12, out=None)

        s_right = torch.squeeze(inputs[1], 2)
        s_right = func.normalize(s_right, p=2, dim=1, eps=1e-12, out=None)
        
        s_right_ = torch.squeeze(inputs[1])

        r = self.sigmoid(self.W_r(s_left) + self.U_r(self.hidden_state)) # output shape: (batch_size, hidden_state)

        cms = self.act(r * self.U(self.hidden_state)) # output shape: (batch_size, hidden_state)

        c = self.act(self.W_c(s_right) + self.U_c(cms)) # output shape: (batch_size, state_dim)
        ums_ = self.h2s(cms) # output shape: (batch_size, state_dim)

        return (s_right_ + c * ums_), cms
    

if __name__ == "__main__":
    print('start testing')
    test_model = TransitionModel(input_dim=4, hidden_dim=100, output_dim=4, batch_size=10)
    test_model
    test_input = [torch.rand(10, 4, 1), torch.rand(10, 4, 1)]
    test_output = test_model(test_input)
    print(test_output[0].shape)
    print(test_output[1].shape)




        





