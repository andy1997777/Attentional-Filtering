import torch
import torch.nn as nn
import math
from AttentionModel import RNNAttention
from ResidualMeasurementModel import ResidualMeasurementModel
from TransitionModel import TransitionModel
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader


class AttentionalFilter(nn.Module):
    def __init__(self, state_dim, measurement_dim, s0, batch_size, 
                 f, h_inv, 
                 transition_model_hidden_dim, 
                 residual_measurement_model_hidden_dim, mask_hidden_dim, measurement_model_dim, 
                 fusion_hidden_dim) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.s0 = s0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s0.to(self.device)
        self.batch_size = batch_size
        self.state = torch.empty_like(s0)
        self.state = self.s0.to(self.device) # shape: (batch_size, state_dim, 1)
        self.f = f
        self.h_inv = h_inv


        # SSM
        self.transition_model_hidden_dim = transition_model_hidden_dim
        self.tm = TransitionModel(input_dim=self.state_dim, hidden_dim=self.transition_model_hidden_dim, 
                                               output_dim=self.state_dim, batch_size=self.batch_size)
        
        self.residual_measurement_model_hidden_dim = residual_measurement_model_hidden_dim
        self.mask_hidden_dim = mask_hidden_dim
        self.measurement_model_dim = measurement_model_dim
        self.rmm = ResidualMeasurementModel(batch_size=self.batch_size, state_dim=self.state_dim, 
                                                                 measurement_dim=self.measurement_dim, 
                                                                 hidden_dim=self.residual_measurement_model_hidden_dim, 
                                                                 model_dim=self.measurement_model_dim, mask_hidden_dim=self.mask_hidden_dim
                                                                )
        
        # Fusion
        self.fusion_hidden_dim = fusion_hidden_dim
        self.af = RNNAttention(state_dim=self.state_dim, measurement_dim=self.measurement_dim, 
                               hidden_dim=self.fusion_hidden_dim, batch_size=self.batch_size)
        

    def init_everything(self, s0):
        self.tm.hidden_state = torch.rand(self.batch_size, self.transition_model_hidden_dim).to(self.device)
        init.xavier_normal_(self.tm.hidden_state)
        self.af.gru1_hidden_state = torch.rand(1, self.batch_size, self.fusion_hidden_dim).to(self.device)
        init.xavier_normal_(self.af.gru1_hidden_state)
        self.af.gru2_hidden_state = torch.rand(1, self.batch_size, self.fusion_hidden_dim).to(self.device)
        init.xavier_normal_(self.af.gru2_hidden_state)
        self.state = s0
    

    def forward(self, inputs):
        measurement = inputs

        sub_estimation1 = self.f(self.state) # shape: (batch_size, state_dim, 1)
        sub_estimation1, self.tm.hidden_state = self.tm([self.state, sub_estimation1]) # sub_estimation1 shape: (batch_size, state_dim)

        # measurement emission
        measurement_ = self.h_inv(torch.reshape(measurement, (self.batch_size, -1, 1)))
        sub_estimation2 = self.rmm([torch.reshape(measurement, (self.batch_size, -1)), torch.reshape(measurement_, (self.batch_size, -1))]) # sub_estimation2 shape: (batch_size, state_dim)

        # attention fusion
        estimation = self.af([sub_estimation1, sub_estimation2])
        self.state = estimation

        return estimation
    
    
    


        
