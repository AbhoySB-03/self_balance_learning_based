from math import comb

from selfbalancebot import SelfBalanceSim
import pybullet as pb
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import matplotlib.pyplot as plt
import time 
from itertools import product


class MRACController:
    def __init__(self,B, dt=0.001):
        # Initialize your MRAC parameters here (e.g., reference model, adaptation gains)
        # self.Am = np.diag([-10, -1, -10, -1])  # Example reference model dynamics
        # self.Bm = np.array([
        #     [1], 
        #     [0],
        #     [0],
        #     [0]])           # Example reference model input
        self.wn1 = 40.0  # Natural frequency for x
        self.zeta1 = 0.8  # Natural frequency for theta
        self.Am = np.array([
            [0, 1],
            [-self.wn1**2, -2*self.zeta1*self.wn1]
        ])

        self.Bm = np.array([[0], [self.wn1**2]])  # Example reference model input
        
        self.gamma_x = np.array([[100.0]])  # Adaptation gain
        self.gamma_r = np.array([[100.0]])
        self.sig_x = np.array([[0.01]])
        self.sig_r = np.array([[0.01]])
        self.sig_theta = np.array([[10]])
        self.xm = np.zeros(2)  # Reference model states
        self.r = np.zeros(1)  # Reference model input

        self.Kx = np.zeros((1,2))
        self.Kr = np.zeros((1,1))
        
        self.dt = dt

        self.phi_d2 = 6
        def num_monomials_fast(n, d1, d2):
            return comb(n + d2, d2) - comb(n + d1 - 1, d1 - 1)
        theta_length = num_monomials_fast(2, 2, self.phi_d2)
        self.theta = np.zeros((theta_length,1))

        self.P = solve_continuous_lyapunov(self.Am.T, -np.eye(self.Am.shape[0]))  # Lyapunov matrix for stability analysis
        print(self.P)
        self.u = np.zeros(1)

        self.B = B  # Input matrix for the actual system

    def update(self, x,r):
        # Update the reference model states
        self.xm_dot = (self.Am @ self.xm + self.Bm @ r) 
        self.xm += self.xm_dot * self.dt
        self.e = x - self.xm  # Tracking error

        
        # Sigma Modification Update Laws
        Kx_dot = -self.gamma_x @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(x) - self.gamma_x @ self.sig_x * np.linalg.norm(self.e) * self.Kx 
        Kr_dot = -self.gamma_r @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(r) - self.gamma_r @ self.sig_r * np.linalg.norm(self.e) * self.Kr
        theta_dot = np.vstack(self.phi(x, r)) @ np.column_stack(self.e) @ self.P @ self.B - self.sig_theta * np.linalg.norm(self.e) * self.theta


        # deadzone_condition = np.linalg.norm(self.e) > 2*np.linalg.norm(self.P)*0.03
        # Kx_dot = -self.gamma_x @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(x) if not deadzone_condition else np.zeros_like(self.Kx)
        # Kr_dot = -self.gamma_r @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(r) if not deadzone_condition else np.zeros_like(self.Kr)
        # theta_dot = np.vstack(self.phi(x, r, d2=self.phi_d2)) @ np.column_stack(self.e) @ self.P @ self.B if not deadzone_condition else np.zeros_like(self.theta)

        self.Kx += Kx_dot * self.dt
        self.Kr += Kr_dot * self.dt
        self.theta += theta_dot * self.dt

        self.u = self.Kx @ x + self.Kr @ r - self.theta.T @ self.phi(x, r)

    def phi(self, x, r):
        n = len(x)
        features = []
        d1 = 2
        d2 = self.phi_d2
        for powers in product(range(d2 + 1), repeat=n):
            total_degree = sum(powers)
            if d1 <= total_degree <= d2:
                term = np.prod(x ** np.array(powers))
                features.append(term)

        return np.array(features)


        
    def phi_old(self, x, r):        
        return np.array([
            x[0]*x[1]**2,
            self.u[0] * x[0]**2,
            x[1]**2 * x[0]**3,
            x[0]**3            
        ])

    def get_control_input(self, x, r):
        # Compute the control input based on the current states and reference model
        return self.Kx @ x + self.Kr @ r - self.theta.T @ self.phi(x, r)
    
class PlotObject:
    def __init__(self, initial_value, max_length = 10000):
        self.max_length = max_length
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.t_start = time.time()

        self.x_data = [0.0]

        # Ensure it's a 1D array
        initial_value = np.array(initial_value).flatten()
        self.dim = len(initial_value)

        # Create storage for each dimension
        self.y_data = [[] for _ in range(self.dim)]
        for i in range(self.dim):
            self.y_data[i].append(initial_value[i])

        # Create one line per dimension
        self.lines = []
        for i in range(self.dim):
            line, = self.ax.plot(self.x_data, self.y_data[i], label=f"State {i}")
            self.lines.append(line)

        self.ax.legend()

    def add_data(self, data):
        t = time.time() - self.t_start
        self.x_data.append(t)

        data = np.array(data).flatten()

        # Handle dimension change safely
        if len(data) != self.dim:
            print(f"[Warning] Dimension changed from {self.dim} to {len(data)}. Skipping.")
            return

        for i in range(self.dim):
            self.y_data[i].append(data[i])

        if len(self.x_data)>self.max_length:
            self.x_data = self.x_data[1:]
            self.y_data = self.y_data[1:]

    def draw(self):

        if len(self.x_data) < 2:
            return

        for i, line in enumerate(self.lines):
            line.set_xdata(self.x_data)
            line.set_ydata(self.y_data[i])

        self.ax.relim()
        self.ax.autoscale_view()

    
# class PlotObject:
#     def __init__(self, initial_value: np.ndarray) -> None:
#         self.fig, self.ax = plt.subplots()
#         self.t_start = time.time()
#         self.x_data = [0.0]
#         self.y_data = np.vstack([np.hstack([initial_value])])
#         self.line = self.ax.plot(self.x_data, self.y_data)
    

#     def add_data(self, data: np.ndarray):
#         self.x_data.append(time.time()-self.t_start)
#         self.y_data = np.vstack([self.y_data, data])
        
#     def draw(self):
#         self.line[0].set_xdata(self.x_data)
#         self.line[0].set_ydata(self.y_data[:0])
#         self.line[1].set_xdata(self.x_data)
#         self.line[1].set_ydata(self.y_data[:1])

#         print(f"X: {len(self.x_data)} Y: {len(self.y_data)}")        
#         self.ax.relim()
#         self.ax.autoscale_view()
#         plt.draw()
#         plt.pause(0.001)
        

class SelfBalanceMRAC(SelfBalanceSim):
    '''
    Self Balance Bot implementation using a simple MRAC controller.
    '''
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        super().Start(start_pos, start_orientation, model_path)
        # Initialize your MRAC parameters here (e.g., reference model, adaptation gains)
        B=np.array([[0],[85.9]])

        self.Controller = MRACController(B=B, dt=self.dt)   # The B matrix is to be corrected.
        self.get_states()
        self.angular_states = self.states[2:]
        self.start_time = time.time()

    def Update(self):
        self.get_states()
        r =np.array([ 0.14*np.exp(-1*(time.time()-self.start_time))*np.sin(time.time()-self.start_time) ]) # Example reference input
        # r = np.array([0])
        # r = np.array([0.15*np.sin(1*(time.time()-self.start_time))])
        self.angular_states = self.states[2:]
        control_input = self.Controller.get_control_input(self.angular_states, r).flatten() # Example reference input
        print(f"Kx: {self.Controller.Kx}, Kr: {self.Controller.Kr}, theta: {self.Controller.theta.T}")  
        print(f"x: {self.states}; input: {control_input}")
        self.apply_input(-control_input[0], cmd_type='torque')  # Apply the control input to the system
        # Debugging statement to check states and control input
        self.Controller.update(self.angular_states, r)  # Update the MRAC controller with the current states and reference
        # print(f"E: {self.angular_states-self.Controller.xm}")

        


    def Quit(self):
        plt.close('all')

if __name__ == "__main__":
     f = SelfBalanceMRAC(del_t=1/240)
     plt.close('all')
     