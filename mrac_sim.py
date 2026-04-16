from selfbalancebot import SelfBalanceSim
import pybullet as pb
import numpy as np
from scipy.linalg import solve_continuous_lyapunov


class MRACController:
    def __init__(self,B, dt=0.001):
        # Initialize your MRAC parameters here (e.g., reference model, adaptation gains)
        self.Am = np.array(
            [[0, 1, 0, 0],
             [-1, -10, 0, 0],
             [0, 0, 0, 1],
             [0, 0, -1, -10]]
        )  # Example reference model dynamics
        self.Bm = np.array([
            [0], 
            [1],
            [0],
            [0]])           # Example reference model input
        
        self.gamma_x = np.array([[10.0]])  # Adaptation gain
        self.gamma_r = np.array([[10.0]])
        self.sig_x = np.array([[0.1]])
        self.sig_r = np.array([[0.1]])
        self.xm = np.zeros(4)  # Reference model states
        self.r = np.zeros(1)  # Reference model input

        self.Kx = np.zeros((1,4))
        self.Kr = np.zeros((1,1))
        self.dt = dt

        self.P = solve_continuous_lyapunov(self.Am.T, -np.eye(4))  # Lyapunov matrix for stability analysis

        self.B = B  # Input matrix for the actual system

    def update(self, x,r):
        # Update the reference model states
        self.xm_dot = (self.Am @ self.xm + self.Bm @ self.r) 
        self.xm += self.xm_dot * self.dt
        self.e = x - self.xm  # Tracking error

        
        # Sigma Modification Update Laws
        Kx_dot = -self.gamma_x @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(x) - self.gamma_x @ self.sig_x * self.Kx
        Kr_dot = -self.gamma_r @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(r) - self.gamma_r @ self.sig_r * self.Kr

        self.Kx += Kx_dot * self.dt
        self.Kr += Kr_dot * self.dt

    def get_control_input(self, x, r):
        # Compute the control input based on the current states and reference model
        return self.Kx @ x + self.Kr @ r

class SelfBalanceMRAC(SelfBalanceSim):
    '''
    Self Balance Bot implementation using a simple MRAC controller.
    '''
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        super().Start(start_pos, start_orientation, model_path)
        # Initialize your MRAC parameters here (e.g., reference model, adaptation gains)
        self.Controller = MRACController(B=np.array([[0],[1],[0],[2]]), dt=self.dt)   # The B matrix is to be corrected.
    def Update(self):
        self.get_states()
        control_input = self.Controller.get_control_input(self.states, np.zeros(1)).flatten() # Example reference input
        print(f"States: {self.Controller.Kx}, Control Input: {control_input}")  
        self.apply_input([control_input[0], control_input[0]], cmd_type='velocity')  # Apply the control input to the system
        # Debugging statement to check states and control input
        self.Controller.update(self.states, np.zeros(1))  # Update the MRAC controller with the current states and reference

if __name__ == "__main__":
     f = SelfBalanceMRAC()