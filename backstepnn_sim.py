import numpy as np
from selfbalancebot import SelfBalanceSim
import pybullet as pb
import time 


def sigmoid(x, derivative=False):
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    return s

class BacksteppingNN:
    def __init__(self, input_dim=2, hidden_size=10, output_dim=1):
        # n=2 for second order, L=hidden_size
        self.L = hidden_size
        self.n = input_dim # [x1, x2]
        
        # V: (input_dim + 1) x hidden_size
        self.V = np.random.standard_normal((self.n + 1, self.L)) * 0.1
        # W: (hidden_size + 1) x output_dim
        self.W = np.random.standard_normal((self.L + 1, output_dim)) * 0.1

    def augmented(self, x):
        # Ensures x is treated as a vector and prepends the bias term 1
        return np.insert(np.atleast_1d(x), 0, 1)

    def forward_parts(self, x_vec):
        """Returns the internal components needed for the update laws."""
        xa = self.augmented(x_vec)
        z = self.V.T @ xa
        sigma = sigmoid(z)
        sigma_aug = self.augmented(sigma)
        sigma_prime = sigmoid(z, derivative=True)
        # sigma_prime_mat is the diagonal matrix of derivatives (diag(sigma'))
        # But in the augmented vector context, we treat it carefully
        return xa, z, sigma_aug, sigma_prime

    def get_f_hat(self, x_vec):
        _, _, sigma_aug, _ = self.forward_parts(x_vec)
        return self.W.T @ sigma_aug

    def compute_control(self, x, x1d, dx1d, ddx1d, c1, c2, g_val):
        """
        Implements the backstepping control law.
        x: [x1, x2]
        x1d: desired x1
        """
        x1, x2 = x[0], x[1]
        
        # Step 1: Virtual Control
        z1 = x1 - x1d
        alpha1 = -c1 * z1 + dx1d
        
        # Step 2: Actual Control
        z2 = x2 - alpha1
        
        # Derivative of alpha1: -c1*(z1_dot) + x1d_ddot
        # z1_dot = x2 - dx1d
        dot_alpha1 = -c1 * (x2 - dx1d) + ddx1d
        
        f_hat = self.get_f_hat(x)
        
        # Backstepping Control Law u
        u = (1.0 / g_val) * (-c2 * z2 - z1 - f_hat + dot_alpha1)
        
        return u, z1, z2
    
    def update_weights(self, x, z1, z2, Gam_w, Gam_v, sig_w, sig_v, dt):
        xa, z, sigma_aug, sigma_prime = self.forward_parts(x)
        
        # 1. Update for W (Output Weights)
        # term_W = (sigma_aug - sigma_prime_augmented * V^T * xa)
        # Note: We need to pad sigma_prime with a 0 at the start for the bias term
        sigma_prime_aug = np.insert(sigma_prime, 0, 0)
        
        # We need the inner product part: sigma_prime * (V.T @ xa)
        # This only applies to the hidden units, not the bias
        v_linear = np.zeros_like(sigma_aug)
        v_linear[1:] = sigma_prime * (self.V.T @ xa)
        
        term_W = sigma_aug - v_linear
        
        # Wdot shape must match self.W (L+1, output_dim)
        # z2 is a scalar or (1,), so we ensure broadcasting works
        Wdot = Gam_w @ (term_W.reshape(-1, 1) * z2 - sig_w * self.W)
        
        # 2. Update for V (Hidden Weights)
        # W_no_bias is (L, output_dim). sigma_prime is (L,)
        W_no_bias = self.W[1:, :] 
        
        # Element-wise multiply W by sigma_prime, then scale by z2
        # result shape: (hidden_size, output_dim)
        backprop_error = z2 * (W_no_bias * sigma_prime.reshape(-1, 1))
        
        # Vdot shape must match self.V (n+1, L)
        # xa is (n+1,), backprop_error.T is (output_dim, L)
        # For single output, backprop_error.T is (1, L)
        Vdot = Gam_v @ (np.outer(xa, backprop_error.sum(axis=1)) - sig_v * self.V)
        
        # Apply updates
        self.W += Wdot * dt
        self.V += Vdot * dt

    def update_weights_prev(self, x, z1, z2, Gam_w, Gam_v, sig_w, sig_v, dt):
        """
        Implements the Taylor-linearized Lyapunov update laws.
        """
        xa, z, sigma_aug, sigma_prime = self.forward_parts(x)
        
        # 1. Update for W (Output Weights)
        # We need (sigma - sigma' * V^T * xa)
        # Note: sigma_prime is for the hidden units, we must align with augmented sigma
        sigma_prime_diag = np.diag(sigma_prime)
        
        # Term: (sigma - sigma' * V.T @ xa)
        # We ignore the bias index 0 for the derivative part
        term_W = sigma_aug.copy()
        term_W[1:] -= sigma_prime_diag @ (self.V.T @ xa)
        
        # Wdot = Gam_w * (term_W * z2 - sig_w * W)
        Wdot = Gam_w @ (term_W.reshape(-1, 1) * z2 - sig_w * self.W)
        
        # 2. Update for V (Hidden Weights)
        # Vdot = Gam_v * (xa * (z2 * W^T * sigma') - sig_v * V)
        # W_no_bias is the W matrix excluding the bias row
        W_no_bias = self.W[1:, :] 
        
        backprop_error = z2 * (W_no_bias @ sigma_prime_diag) # shape (1, hidden_size)
        Vdot = Gam_v @ (np.outer(xa, backprop_error) - sig_v * self.V)
        
        # Apply updates
        self.W += Wdot * dt
        self.V += Vdot * dt


class SelfBalanceBNN(SelfBalanceSim):
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        super().Start(start_pos, start_orientation, model_path)
        self.Controller = BacksteppingNN(2, 10, 1)

    def g(self, x):
        """
        Calculates the g4 component of the control vector g(x) for the self_balance_bot.
        This represents the mapping from total motor torque to angular acceleration (theta_ddot).
        
        Parameters:
        x3 (float): The tilt angle (theta) in radians.
        
        Returns:
        float: The value of the g4 term.
        """
        # Physical constants derived from URDF
        Mt = 1.75185    # Total effective inertial mass
        It = 0.09381    # Total body inertia about the axle
        Ml = 0.23648    # Mass * distance to center of mass
        r = 0.20006     # Wheel radius
        
        # Pre-calculate trigonometric values
        cos_x3 = np.cos(x)
        
        # Expanded Numerator: -Mt - (Ml/r)*cos(x3)
        numerator = -Mt - (Ml / r) * cos_x3
        
        # Expanded Denominator (Determinant D): Mt*It - (Ml*cos(x3))^2
        denominator = (Mt * It) - (Ml * cos_x3)**2
        
        return numerator / denominator

        
    def Update(self):
        self.get_states()
        _,_,x1,x2 = self.states
        x = np.array([x1, x2])

        input, e1, e2 = self.Controller.compute_control(x, 0, 0, 0, 40, 40,self.g(x1))

        self.apply_input(input, 'velocity')

        self.Controller.update_weights(x, e1, e2, np.eye(11)*30, np.eye(3)*30, 0.01, 0.01, self.dt)

if __name__=='__main__':
    f=SelfBalanceBNN(del_t=1/1000)
    
    