import torch.utils.data as data
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from random import randint
from torch.utils.data import DataLoader
from generate_data import Lorenz
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from AttentionalFilter import AttentionalFilter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def __plot_trajectory(states):
    fig = plt.figure(linewidth=0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.8)
    
    # plt.axis('off')
    plt.tight_layout()
    
    plt.show()


def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader):
    model.eval()
    result=torch.zeros((1, 5000, 3, 1))
    loss_fn = nn.MSELoss(reduction='mean')
    epoch_MSE_loss = 0
    for batch_idx, (meas, state) in enumerate(test_loader):
        meas = torch.unsqueeze(meas, dim=-1).to(DEVICE)
        state = torch.unsqueeze(state, dim=-1).to(DEVICE)
        MSE_loss = 0
        model.init_everything(meas[:, 0, :, :])
        for i in range(meas.shape[1]):
            est_state = model(meas[:, i, :, :])
            result[0, i, :, :] = est_state
            MSE_loss += loss_fn(est_state, state[:, i, :, :])
            
        
        MSE_loss = MSE_loss / meas.shape[1] 
        epoch_MSE_loss += (MSE_loss.item())


    print('Test MSE:', 10*np.log10(epoch_MSE_loss / len(test_loader)))
    return epoch_MSE_loss / len(test_loader), result



def train_model(model, optimizer, train_loader, epochs):
    loss_fn = nn.MSELoss(reduction='mean')
    # 开始训练
    for e in tqdm(range(epochs)):
        epoch_MSE_loss = 0
        # TRAIN
        model.train()
        for batch_idx, (meas, state) in enumerate(train_loader):
            meas = torch.unsqueeze(meas, dim=-1).to(DEVICE)
            state = torch.unsqueeze(state, dim=-1).to(DEVICE)

            MSE_loss = 0

            model.init_everything(meas[:, 0, :, :])
            for i in range(meas.shape[1]):
                est_state = model(meas[:, i, :, :])
                MSE_loss +=  loss_fn(est_state, state[:, i, :, :])
                 
            MSE_loss = MSE_loss / meas.shape[1]

            optimizer.zero_grad()
            MSE_loss.backward()
            optimizer.step()

            epoch_MSE_loss += (MSE_loss.item())
    
        print()
        print('Train MSE:', 10*np.log10(epoch_MSE_loss / len(train_loader)))



if __name__=="__main__":
    batch_size = 50
    test_batch_size = 1
    J = 5


    def f(x, J=J, delta_t=0.01):
        C = torch.tensor([[-10, 10, 0],
                      [28, -1, 0],
                      [0, 0, -8 / 3]]).float().to(DEVICE)
        batch_size = x.shape[0]
        BX = torch.zeros([batch_size, 3, 3]).float().to(DEVICE)
        BX[:,1,0] = torch.squeeze(-x[:,2,:])
        BX[:,2,0] = torch.squeeze(x[:,1,:])
        A = torch.add(BX, C)
        # Taylor Expansion for F    
        F = torch.eye(3).to(DEVICE)
        F = F.reshape((1, 3, 3)).repeat(batch_size, 1, 1) # [batch_size, state_dim, state_dim]
        for j in range(1, J+1):
            F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
            F = torch.add(F, F_add)
        return torch.bmm(F, x)
    
    def h_inv(x):
        return x
    

    train_num = 500
    test_num = 50
    dataset_train = Lorenz(partition='train', sample_dt=0.01, max_len=100, test_tt=test_num*100, val_tt=test_num*100, tr_tt=train_num*100)
    dataset_test = Lorenz(partition='test', sample_dt=0.01, max_len=test_num*100, test_tt=test_num*100, val_tt=test_num*100, tr_tt=train_num*100)
    dataset_val = Lorenz(partition='val', sample_dt=0.01, max_len=100, test_tt=test_num*100, val_tt=test_num*100, tr_tt=train_num*100)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)


    # 创建模型
    x_0 = torch.tensor([[1.0], 
                        [1.0],
                        [1]]).to(DEVICE)
    x_0_batch = x_0.unsqueeze(0).expand(batch_size, -1, -1)

    af = AttentionalFilter(state_dim=3, measurement_dim=3, s0=x_0_batch, batch_size=batch_size, 
                 f=f, h_inv=h_inv, 
                 transition_model_hidden_dim=32, 
                 residual_measurement_model_hidden_dim=32, mask_hidden_dim=32, measurement_model_dim=3, 
                 fusion_hidden_dim=32)

    # 训练参数
    best_val = 1e8
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-5
    optimizer_af = optim.Adam(af.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # 开始训练
    print('----------- train -----------')
    train_model(model=af, optimizer=optimizer_af, train_loader=train_loader, epochs=epochs)
    
    # 开始测试
    print('----------- test -----------')
    af.batch_size = 1
    af.tm.batch_size = 1
    af.rmm.batch_size = 1
    af.af.batch_size = 1

    _, result = test_model(model=af, test_loader=test_loader)
    result = torch.squeeze(result).detach().numpy()
    print(result.shape)
    __plot_trajectory(result)


    