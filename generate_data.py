import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch.utils.data as data
import torch
from mpl_toolkits.mplot3d import Axes3D
import math
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

### Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = torch.tensor([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)

def h_gen_R(x):
    H = RotMatrix @ torch.eye(3)
    return H @ x


class MeasurementModel():
    def __init__(self, H, R):
        self.H = H
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = self.H @ x + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement
    

class Lorenz(data.Dataset):
    def __init__(self, partition='train', max_len=1000, tr_tt=1000, val_tt=1000, test_tt=1000, sample_dt=0.05, 
                 lamb=0.5):
        self.partition = partition  # training set or test set
        self.max_len = max_len
        self.lamb = np.sqrt(lamb)
        self.x0 = [1.0, 1.0, 1.0]
        self.H = np.diag([1]*3)
        self.R = np.diag([1]*3) * self.lamb ** 2
        self.sample_dt = sample_dt
        self.dt = 0.00001
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0

        self.data = self._generate_sample(seed=0, tt=test_tt+val_tt+tr_tt)

        if self.partition == 'test':
            self.data = [self.data[0][0:test_tt], self.data[1][0:test_tt]]
        elif self.partition == 'val':
            self.data = [self.data[0][test_tt:(test_tt+val_tt)], self.data[1][test_tt:(test_tt+val_tt)]]
        elif self.partition == 'train':
            self.data = [self.data[0][(test_tt+val_tt):(test_tt + val_tt + tr_tt)], self.data[1][(test_tt+val_tt):(test_tt + val_tt + tr_tt)]]
        else:
            raise Exception('Wrong partition')
        self._split_data()
    
    def _generate_sample(self, seed, tt):
        np.random.seed(seed)
        sample = self._simulate_system(tt=tt, x0=self.x0)
        return list(sample)

    def f(self, state, t):
        x, y, z = state  # unpack the state vector
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  # derivatives
    
    def _simulate_system(self, tt, x0):
        t = np.arange(0.0, tt*self.sample_dt, self.dt)
        states = odeint(self.f, x0, t)
        states_ds = np.zeros((tt, 3))
        for i in range(states_ds.shape[0]):
            states_ds[i] = states[i*int(self.sample_dt/self.dt)]
        states = states_ds # shape: (tt, 3)

        #Measurement
        meas_model = MeasurementModel(self.H, self.R)
        meas = np.zeros(states.shape)
        for i in range(len(states)):
            meas[i] = meas_model(states[i])
        return states, meas

    def _split_data(self):
        num_splits = math.ceil(float(self.data[0].shape[0])/self.max_len)
        data = []
        for i in range(int(num_splits)):
            i_start = i*self.max_len
            i_end = (i+1)*self.max_len
            data.append([self.data[0][i_start:i_end], self.data[1][i_start:i_end]])
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        state, meas = self.data[index]
        state = torch.from_numpy(state)
        meas = torch.from_numpy(meas)
        state = state.to(torch.float32)
        meas = meas.to(torch.float32)
        state0 = meas[0, :]
        return meas, state



def __plot_trajectory(states):
    fig = plt.figure(linewidth=0.0)
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.5)
    plt.axis('off')
    plt.show()


 
if __name__=='__main__':
    train_num = 500
    test_num = 50

    dataset_train = Lorenz(partition='train', sample_dt=0.01, max_len=100, test_tt=test_num*100, val_tt=test_num*100, tr_tt=train_num*100)
    dataset_test = Lorenz(partition='test', sample_dt=0.01, max_len=test_num*100, test_tt=test_num*100, val_tt=test_num*100, tr_tt=train_num*100)
    # dataset_val = Lorenz(partition='val', sample_dt=0.01, max_len=100, test_tt=test_num*100, val_tt=test_num*100, tr_tt=train_num*100)
    
    print(len(dataset_train))
    state, meas = dataset_test.data[0]
    print(state.shape)
    __plot_trajectory(state)
    __plot_trajectory(meas)


    

