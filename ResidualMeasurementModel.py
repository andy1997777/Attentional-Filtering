import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as func


class ResidualMeasurementModel(nn.Module):
    def __init__(self, batch_size, state_dim, measurement_dim, hidden_dim, model_dim, mask_hidden_dim) -> None:
        super().__init__()
        
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.mask_hidden_dim = mask_hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = nn.Tanh()

        # estimating sub-state with DNN from orignal observiation 
        self.trunk_func = nn.Sequential(nn.Linear(self.measurement_dim, self.hidden_dim), 
                                        self.act, 
                                        nn.Linear(self.hidden_dim, self.hidden_dim), 
                                        self.act, 
                                        nn.Linear(self.hidden_dim, self.hidden_dim), 
                                        self.act, 
                                        nn.Linear(self.hidden_dim, self.state_dim))

        # soft mask 
        self.m1 = nn.Parameter(torch.empty(self.mask_hidden_dim, self.model_dim))
        self.m2 = nn.Parameter(torch.empty(self.model_dim, self.mask_hidden_dim))

        self.b1 = nn.Parameter(torch.empty(self.mask_hidden_dim, 1))
        self.b2 = nn.Parameter(torch.empty(self.model_dim, 1))


        self.initialize_parameters()

    def initialize_parameters(self):
        # 使用 Xavier 均匀初始化模型中的所有参数
        for param in self.parameters():
            if param.dim() > 1:
                init.xavier_normal_(param)
                # init.xavier_uniform_(param)

    
    def forward(self, inputs):
        """
        inputs: list[orignal observation(batch_size, measurement_dim); z(batch_size, model_dim), ]
        """

        s_left = inputs[0] # orignal measurement
        s_left = func.normalize(s_left, p=2, dim=1, eps=1e-12, out=None)
        s_right = inputs[1]
        
        s_left = self.trunk_func(s_left)
        
        # soft mask 
        mask_input = s_left - s_right
        mask_input = mask_input / s_right
        mask_input = mask_input * mask_input # output shape (batch_size, model_dim)
        mask_input = torch.reshape(mask_input, (self.batch_size, -1, 1)) # output shape (batch_size, model_dim, 1)
        batch_m1 = torch.unsqueeze(self.m1 * self.m1, 0).expand(self.batch_size, -1 , -1)
        batch_b1 = torch.unsqueeze(self.b1 * self.b1, 0).expand(self.batch_size, -1 , -1)
        mask_output = torch.bmm(batch_m1, mask_input) + batch_b1 # output shape[batch_size, mask_hidden_dim, 1]
        mask_output = self.act(mask_output)
        batch_m2 = torch.unsqueeze(self.m2 * self.m2, 0).expand(self.batch_size, -1 , -1)
        batch_b2 = torch.unsqueeze(self.b2 * self.b2, 0).expand(self.batch_size, -1 , -1)
        mask_output = torch.bmm(batch_m2, mask_output) + batch_b2  # output shape[batch_size, model_dim, 1]
        mask_output = self.act(mask_output)

        mask_output = torch.reshape(mask_output, (self.batch_size, -1))

        result = torch.zeros((self.batch_size, self.state_dim))

        result = mask_output * s_left + s_right
        


        return result
    

if __name__ == "__main__":
    print('start testing')
    test_model = ResidualMeasurementModel(batch_size=10, state_dim=5, measurement_dim=5, hidden_dim=100, model_dim=5, mask_hidden_dim=50)
    test_model
    test_input = [torch.rand(10, 5), torch.rand(10, 5)]
    test_output = test_model(test_input)
    print(test_output.shape)







