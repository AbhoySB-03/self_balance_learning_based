import time

from selfbalancebot import SelfBalanceSim
import pybullet as pb
import numpy as np
from scipy.linalg import solve_continuous_lyapunov


class MRACController:
    def __init__(self,B, dt=0.001):
        # Initialize your MRAC parameters here (e.g., reference model, adaptation gains)
        self.wn1 = 10.0  # Natural frequency for x
        self.zeta1 = 0.9 # Natural frequency for theta
        self.Am = np.array([
            [0, 1, 0, 0],
            [-self.wn1**2, -2*self.zeta1*self.wn1, 0, 0],
            [0, 0, -10, 0],
            [0, 0, -0, -1]
        ])          
        
        self.Bm = np.array([[0], [self.wn1**2], [0], [0]])  # Example reference model input


        # self.wn1 = 5.0  # Natural frequency for x
        # self.zeta1 = 1.7  # Natural frequency for theta
        # self.Am = np.array([
        #     [0, 1, 0, 0],
        #     [-self.wn1**2, -2*self.zeta1*self.wn1, 0, 0],
        #     [0, 0, -10, 0],
        #     [0, 0, -0, -1]
        # ])
        # self.Bm = np.array([[0], [self.wn1**2], [0], [0]])  # Example reference model input
        
        self.gamma_x = np.array([[5.0]])  # Adaptation gain
        self.gamma_r = np.array([[5.0]])
        self.sig_x = np.array([[10.0]])
        self.sig_r = np.array([[10.0]])
        self.xm = np.zeros(4)  # Reference model states
        self.r = np.zeros(1)  # Reference model input

        self.Kx = np.zeros((1,4))
        self.Kr = np.zeros((1,1))
        self.dt = dt

        self.P = solve_continuous_lyapunov(self.Am.T, -np.eye(4))  # Lyapunov matrix for stability analysis
        print(self.P)

        self.B = B  # Input matrix for the actual system

    def update(self, x,r):
        # Update the reference model states
        self.xm_dot = (self.Am @ self.xm + self.Bm @ r) 
        self.xm += self.xm_dot * self.dt
        self.e = x - self.xm  # Tracking error

        
        # Sigma Modification Update Laws
        Kx_dot = -self.gamma_x @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(x) - self.gamma_x @ self.sig_x * np.linalg.norm(self.e)*self.Kx
        Kr_dot = -self.gamma_r @ self.B.T @ self.P @ np.vstack(self.e) @ np.column_stack(r) - self.gamma_r @ self.sig_r * np.linalg.norm(self.e)*self.Kr

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
        B=np.array([[0],[12.54],[0],[85.9]])

        self.Controller = MRACController(B=B, dt=self.dt)   # The B matrix is to be corrected.

    def yaw_pid(self, target_yaw=0.0):
        # Simple P controller for yaw control
        _, orn = pb.getBasePositionAndOrientation(self.model)
        _, ang_vel = pb.getBaseVelocity(self.model) 
        euler = pb.getEulerFromQuaternion(orn)
        current_yaw = euler[2]
        current_yaw_rate = ang_vel[2]
        error = target_yaw - current_yaw 
        Kp = 1.0  # Proportional gain (tune as needed)
        Kd = 0.5   # Derivative gain (tune as needed)
        return Kp * error - Kd * current_yaw_rate
    
    def Update(self):
        self.get_states()
        r = [0]  # Example reference input
        control_input = self.Controller.get_control_input(self.states, r).flatten() # Example reference input
        # print(f"Kx: {self.Controller.Kx}, Kr: {self.Controller.Kr}, Control Input: {control_input}")  
        self.apply_input(-control_input[0], cmd_type='torque')  # Apply the control input to the system
        # Debugging statement to check states and control input
        self.Controller.update(self.states, r)  # Update the MRAC controller with the current states and reference

if __name__ == "__main__":
     f = SelfBalanceMRAC(del_t=1/240)