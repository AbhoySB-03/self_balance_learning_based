import pybullet as pb
from selfbalancebot import SelfBalanceSim
import numpy as np


def sigmoid(x):
    # Example activation function (ReLU)
    return 1 / (1 + np.exp(-x))

def sigm_dash(x):
    s = sigmoid(x)
    return s * (1 - s)

class NNControlller:
    def __init__(self, n, L, m, activation):
        # Initialize your neural network here (e.g., load weights, define architecture)
        self.n = n
        self.L = L
        self.m = m
        self.activation = activation
        # Example: Initialize weights for a simple feedforward network
        self.W = np.random.randn(L+1, m)
        self.V = np.random.randn(n+1, L)

    def augment(self, x):
        # Add bias term to the input
        return np.append([1], x)
    
    def forward(self, x):
        # Forward pass through the neural network
        x_aug = self.augment(x)
        h = self.activation(np.dot(self.V.T, x_aug))
        h_aug = self.augment(h)
        output = np.dot(self.W.T, h_aug)
        print(f"NN Output: {output}")  # Debugging statement to check the output of the NN
        return output
    
    def update_weights(self, x, e, gamma_w, gamma_v, sig_w=1, sig_v=1, dt=0.001):
        # Update weights based on the error (this is a placeholder, implement your learning rule)
        Wdot = gamma_w @ self.activation(np.dot(self.V, self.augment(x)) @ self.hstack(e)) - gamma_w @ sig_w * self.W
        Vdot = gamma_v @ self.augment(x) @ self.hstack(e) @ self.W.T @ self.activation(np.dot(self.V, self.augment(x)))
        - gamma_v @ sig_v * self.V 

        self.W += Wdot * dt
        self.V += Vdot * dt

    def get_control_input(self, e, xd_dot, x,K):
        # Use the states to compute the control input using your neural network
        return -K @ e - self.forward(x) + xd_dot
        
        
class SelfBalanceNN(SelfBalanceSim):
    '''
    Self Balance Bot implementation using a simple Neural Network controller.
    '''
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        super().Start(start_pos, start_orientation, model_path)
        # Initialize your neural network here (e.g., load weights, define architecture)
        self.nn_controller = NNControlller(n=4, L=10, m=4, activation=sigmoid)

    def Update(self):
        self.get_states()
        # Use the states to compute the control input using your neural network
        # For example:
        # input_to_nn = self.states
        # wheel_vel = your_neural_network(input_to_nn)
        # self.apply_input([wheel_vel, wheel_vel], cmd_type='velocity')
        xd=np.zeros(4)  # Desired state (you can set this to your desired values)
        e = self.states - xd
        xd_dot = np.zeros(4)  # Desired state derivative (you can set this to your desired values)
        K = np.diag([1,1,1,1])  # Gain matrix (you can
        
        u = self.nn_controller.get_control_input(e, xd_dot, self.states, K)
        self.apply_input(u, cmd_type='velocity')    

        self.nn_controller.update_weights(self.states, e, gamma_w=np.eye(2), gamma_v=np.eye(10), dt=self.dt)

if __name__ == "__main__":
     f = SelfBalanceNN()