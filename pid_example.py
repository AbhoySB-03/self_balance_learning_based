from selfbalancebot import SelfBalanceSim
import pybullet as pb
import numpy as np

class SelfBalancePID(SelfBalanceSim):
    '''
    Self Balance Bot implementation using a simple PID controller.
    '''
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        super().Start(start_pos, start_orientation, model_path)
        self.Kp = [100, 1]
        self.Kd = [1, 1]

    def Update(self):
        self.get_states()
        x, x_dot,theta, theta_dot = self.states
        wheel_vel = self.Kp[0]*theta - self.Kd[0]*theta_dot - self.Kd[1]*x_dot
        self.apply_input(wheel_vel, cmd_type='velocity')

    def PostUpdate(self):
        # Print states for debugging
        print(f"States: {self.states}")

if __name__ == "__main__":
     f = SelfBalancePID()
