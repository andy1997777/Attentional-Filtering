import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader


class RNNAttention(nn.Module):
    def __init__(self, state_dim, measurement_dim, hidden_dim, batch_size) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.act = nn.ReLU()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # global neural network
        self.g1 = nn.Parameter(torch.empty(self.hidden_dim, self.state_dim))

        # local neural network
        self.l1 = nn.Parameter(torch.empty(self.hidden_dim, self.state_dim))

        # build neural networks
        self.gru1 = nn.GRU(self.hidden_dim, self.hidden_dim, 1, batch_first=True)
        self.gru2 = nn.GRU(self.hidden_dim, self.hidden_dim, 1, batch_first=True)
        
        self.gru1_hidden_state = torch.rand(1, self.batch_size, self.hidden_dim).to(self.device)
        self.gru2_hidden_state = torch.rand(1, self.batch_size, self.hidden_dim).to(self.device)

        self.scoring_sequential = nn.Sequential(
                                                nn.Linear(self.hidden_dim, self.hidden_dim*2), 
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim*2, self.state_dim)
                                                )
        
        self.initialize_parameters()
    

    def initialize_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                init.xavier_normal_(param)
                # init.xavier_uniform_(param)
    

    def forward(self, sub_estimation):

        sub_estimation1 = sub_estimation[0] # shape: (batch_size, state_dim)
        sub_estimation2 = sub_estimation[1] # shape: (batch_size, state_dim)

        sub_estimation1 = torch.reshape(sub_estimation1, (self.batch_size, -1, 1))
        sub_estimation2 = torch.reshape(sub_estimation2, (self.batch_size, -1, 1))

        # global information
        input_feature = torch.cat((sub_estimation1, sub_estimation2), dim=2) # shape [batch_size, state_dim, 2]
        global_information = torch.mean(input_feature, dim=2) # shape: (batch_size, state_dim)
        global_information = torch.reshape(global_information, (self.batch_size, self.state_dim, 1))
        global_information = func.normalize(global_information, p=2, dim=1, eps=1e-12, out=None)
        
        # local information; sub-estimations themselves; shape [batch_size, state_dim]
        local_information1 = sub_estimation1 # shape [batch_size, state_dim]
        local_information1 = torch.reshape(local_information1, (self.batch_size, self.state_dim, 1))
        local_information1 = func.normalize(local_information1, p=2, dim=1, eps=1e-12, out=None)

        local_information2 = sub_estimation2 # shape [batch_size, state_dim]
        local_information2 = torch.reshape(local_information2, (self.batch_size, self.state_dim, 1))
        local_information2 = func.normalize(local_information2, p=2, dim=1, eps=1e-12, out=None)

        batch_g1 = torch.unsqueeze(self.g1, 0).expand(self.batch_size, -1 , -1)
        batch_l1 = torch.unsqueeze(self.l1, 0).expand(self.batch_size, -1 , -1)

        f1 = torch.bmm(batch_g1, global_information) + torch.bmm(batch_l1, local_information1) # shape:[batch_size, hidden_dim, 1]
        f1 = self.act(f1) # shape:[batch_size, hidden_dim, 1]
        f1 = torch.reshape(f1, (self.batch_size, 1, self.hidden_dim))

        f2 = torch.bmm(batch_g1, global_information) + torch.bmm(batch_l1, local_information2) # shape:[batch_size, hidden_dim, 1]
        f2 = self.act(f2) # shape:[batch_size, hidden_dim, 1]
        f2 = torch.reshape(f2, (self.batch_size, 1, self.hidden_dim))

        e1, self.gru1_hidden_state = self.gru1(f1, self.gru1_hidden_state)
        e2, self.gru2_hidden_state = self.gru2(f2, self.gru2_hidden_state)


        e1 = self.scoring_sequential(torch.squeeze(e1)) # shape [batch_size, state_dim]
        e2 = self.scoring_sequential(torch.squeeze(e2)) # shape [batch_size, state_dim]

        e1 = torch.reshape(e1, (self.batch_size, self.state_dim, 1))
        e2 = torch.reshape(e2, (self.batch_size, self.state_dim, 1))

        e = torch.cat((e1, e2), dim=2) # shape [batch_size, state_dim, 2]
        a = torch.softmax(e, dim=2)

        a1 = a[:, :, 0]
        a2 = a[:, :, 1]
        return a1 * torch.squeeze(sub_estimation1) + a2 * torch.squeeze(sub_estimation2)
    

if __name__ == "__main__":
    print('start testing')
    test_model = RNNAttention(state_dim=3, measurement_dim=3, hidden_dim=20, batch_size=10)
    test_input = [torch.rand(10, 3), torch.rand(10, 3)]
    test_output = test_model(test_input)
    print(test_output)
