from selfbalancebot import SelfBalanceSim
import pybullet as pb
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

class NNMRACController:
    def __init__(self, B, dt=0.001):
        self.dt = dt
        self.B = B  # Input matrix (4x1)
        
        # --- Reference Model ---
        self.Am = np.array([
            [0, 1, 0, 0],
            [-1, -10, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, -10]
        ])
        self.Bm = np.array([[0], [1], [0], [0]])
        self.xm = np.zeros((4, 1))  
        
        # --- Linear MRAC Parameters ---
        self.gamma_x = 50.0 * np.eye(4)  
        self.gamma_r = 2.0 * np.eye(1)
        self.sig_x = 0.1
        self.sig_r = 0.1
        
        self.Kx = np.zeros((1, 4))
        self.Kr = np.zeros((1, 1))
        
        # Continuous Lyapunov equation: Am.T * P + P * Am = -Q
        self.P = solve_continuous_lyapunov(self.Am.T, -np.eye(4))
        
        # --- Neural Network Parameters (RBF) ---
        self.num_nodes = 25 
        self.gamma_w = 2.0 * np.eye(self.num_nodes) 
        self.sig_w = 0.05  
        self.W_hat = np.zeros((self.num_nodes, 1))
        
        # Deterministic RBF Grid for [theta, theta_dot]
        # Concentrating nodes around the equilibrium (0,0)
        theta_centers = np.linspace(-0.5, 0.5, 5)      # +/- ~28 degrees
        theta_dot_centers = np.linspace(-2.0, 2.0, 5)
        grid_th, grid_thd = np.meshgrid(theta_centers, theta_dot_centers)
        
        self.centers = np.zeros((self.num_nodes, 4))
        self.centers[:, 0] = grid_th.flatten()
        self.centers[:, 1] = grid_thd.flatten()
        self.width = 0.5  # Narrower spread for tighter grid

    def compute_rbf(self, x):
        Phi = np.zeros((self.num_nodes, 1))
        for i in range(self.num_nodes):
            dist_sq = np.linalg.norm(x.flatten() - self.centers[i, :])**2
            Phi[i, 0] = np.exp(-dist_sq / (2 * self.width**2))
        return Phi

    # def get_control_input(self, x, r):
    #     x_vec = np.array(x).reshape((4, 1))
    #     r_vec = np.array(r).reshape((1, 1))
        
    #     Phi = self.compute_rbf(x_vec)
        
    #     u_linear = self.Kx @ x_vec + self.Kr @ r_vec
    #     u_nn = self.W_hat.T @ Phi  
        
    #     return (u_linear - u_nn).flatten()

    def get_control_input(self, x, r):
        x_vec = np.array(x).reshape((4, 1))
        r_vec = np.array(r).reshape((1, 1))
        
        Phi = self.compute_rbf(x_vec)
        
        u_linear = self.Kx @ x_vec + self.Kr @ r_vec
        u_nn = self.W_hat.T @ Phi  
        
        # Calculate raw control
        u_raw = (u_linear - u_nn).flatten()
        
        # SATURATE THE INPUT: Limit to max +/- 15 rad/s (approx 140 RPM)
        u_clipped = np.clip(u_raw, -15.0, 15.0) 
        
        return u_clipped

    def update(self, x, r):
        x_vec = np.array(x).reshape((4, 1))
        r_vec = np.array(r).reshape((1, 1))
        
        # 1. Reference Model Update
        xm_dot = self.Am @ self.xm + self.Bm @ r_vec
        self.xm += xm_dot * self.dt
        
        # 2. Tracking Error
        e = x_vec - self.xm 
        
        # 3. Scalar Projection Term (Crucial for matrix math)
        ePB = float((e.T @ self.P @ self.B)[0, 0])
        
        # 4. Update Laws with Sigma Modification
        Kx_dot = -self.gamma_x @ x_vec * ePB - self.sig_x * self.gamma_x @ self.Kx.T
        self.Kx += Kx_dot.T * self.dt
        
        Kr_dot = -self.gamma_r @ r_vec * ePB - self.sig_r * self.gamma_r @ self.Kr.T
        self.Kr += Kr_dot.T * self.dt
        
        Phi = self.compute_rbf(x_vec)
        W_hat_dot = self.gamma_w @ Phi * ePB - self.sig_w * self.gamma_w @ self.W_hat
        self.W_hat += W_hat_dot * self.dt

class SelfBalanceMRAC(SelfBalanceSim):
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        super().Start(start_pos, start_orientation, model_path)
        
        # Corrected B Matrix approximation for velocity-commanded inverted pendulum
        # Adjust the [10.0] term based on your specific wheel-radius to pitch-coupling
        B_matrix = np.array([[0], [10.0], [0], [1.0]])
        self.Controller = NNMRACController(B=B_matrix, dt=self.dt)  
        
    def Update(self):
        self.get_states()
        
        control_input = self.Controller.get_control_input(self.states, np.zeros(1)) 
        
        # Apply identical velocity to both wheels
        self.apply_input([control_input[0], control_input[0]], cmd_type='velocity')  
        
        # Execute adaptation step
        self.Controller.update(self.states, np.zeros(1))

if __name__ == "__main__":
     f = SelfBalanceMRAC()