from pybulletsim import PyBulletSimulation
import pybullet as pb
import numpy as np

class SelfBalanceSim(PyBulletSimulation):
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        print("Started...")
        pb.loadURDF("plane.urdf")
        self.model = pb.loadURDF(model_path,
                                 basePosition=start_pos, baseOrientation=start_orientation)
        n=pb.getNumJoints(self.model)
        self.rev_jnt_ind = []
        self.states = np.zeros(4) # [theta, theta_dot, x, x_dot]
        for i in range(n):
            js=pb.getJointInfo(self.model, i)
            if js[2] == pb.JOINT_REVOLUTE:
                self.rev_jnt_ind.append(i)
        self.apply_input([0,0], cmd_type='torque')

    def get_states(self):
        pos, orn = pb.getBasePositionAndOrientation(self.model)
        euler = pb.getEulerFromQuaternion(orn)
        self.states[0] = euler[0]  # theta
        self.states[2] = pos[1]    # x
        
        lin_vel, ang_vel = pb.getBaseVelocity(self.model)
        self.states[1] = ang_vel[0]  # theta_dot
        self.states[3] = lin_vel[1]  # x_dot

    def apply_input(self, commands, cmd_type='torque'):
        if cmd_type == 'torque':
            pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                         jointIndices=self.rev_jnt_ind,
                                         controlMode=pb.TORQUE_CONTROL,
                                         forces=commands)
        elif cmd_type == 'velocity':
             pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                     jointIndices=self.rev_jnt_ind,
                                     controlMode=pb.VELOCITY_CONTROL,
                                     targetVelocities=commands)

    def apply_input_spec(self, jnt_indices,targetWheelVelocities: np.ndarray, cmd_type='torque'):
        if cmd_type == 'torque':
            pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                         jointIndices=jnt_indices,
                                         controlMode=pb.TORQUE_CONTROL,
                                         forces=targetWheelVelocities)
        elif cmd_type == 'velocity':
            pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                     jointIndices=jnt_indices,
                                     controlMode=pb.VELOCITY_CONTROL,
                                     targetVelocities=targetWheelVelocities)
        
    
if __name__ == "__main__":
     f = SelfBalanceSim()
