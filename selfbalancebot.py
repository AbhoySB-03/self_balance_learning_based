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
                pb.setJointMotorControl2(
                    bodyUniqueId=self.model,
                    jointIndex=i,
                    controlMode=pb.VELOCITY_CONTROL,
                    force=0
                )
        self.input_limit = 40

    def follow_cam(self):
        keys = pb.getKeyboardEvents()

        # Space bar = ASCII 32
        if 32 in keys:
            if keys[32] & pb.KEY_IS_DOWN:
                cam_info = pb.getDebugVisualizerCamera()

                distance = cam_info[10]
                yaw = cam_info[8]
                pitch = cam_info[9]
                print("Following robot")
                pos, _ = pb.getBasePositionAndOrientation(self.model)
                pb.resetDebugVisualizerCamera(cameraDistance = distance,cameraYaw=yaw, 
                                    cameraPitch=pitch,  cameraTargetPosition=pos)
        

    def get_states(self):
        pos, orn = pb.getBasePositionAndOrientation(self.model)

        euler = pb.getEulerFromQuaternion(orn)
        self.states[2] = euler[0]  # theta
        self.states[0] = pos[1]    # x
        
        lin_vel, ang_vel = pb.getBaseVelocity(self.model)
        self.states[3] = ang_vel[0]  # theta_dot
        self.states[1] = lin_vel[1]  # x_dot

    def apply_input(self, command, cmd_type='velocity'):
        command = max(min(command,self.input_limit), -self.input_limit)
        if cmd_type == 'torque':
            pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                         jointIndices=self.rev_jnt_ind,
                                         controlMode=pb.TORQUE_CONTROL,
                                         forces=[command, command])
        elif cmd_type == 'velocity':
             pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                     jointIndices=self.rev_jnt_ind,
                                     controlMode=pb.VELOCITY_CONTROL,
                                     targetVelocities=[command,command])
    
    def apply_input_ik(self, forward: float, yaw=0.0, cmd_type='velocity'):
        wl, wr = self.inverse_kinematics(forward, yaw)  # Assuming no angular velocity for simplicity
        # print(f"Applying command: {command}, Wheel Speeds -> Left: {wl}, Right: {wr}")
        if cmd_type == 'torque':
            pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                         jointIndices=self.rev_jnt_ind,
                                         controlMode=pb.TORQUE_CONTROL,
                                         forces=[wl, wr])
        elif cmd_type == 'velocity':
             pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                     jointIndices=self.rev_jnt_ind,
                                     controlMode=pb.VELOCITY_CONTROL,
                                     targetVelocities=[wl,wr])

    def apply_input_spec(self, jnt_indices,commands: np.ndarray, cmd_type='torque'):
        if cmd_type == 'torque':
            pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                         jointIndices=jnt_indices,
                                         controlMode=pb.TORQUE_CONTROL,
                                         forces=commands)
        elif cmd_type == 'velocity':
            pb.setJointMotorControlArray(bodyUniqueId=self.model,
                                     jointIndices=jnt_indices,
                                     controlMode=pb.VELOCITY_CONTROL,
                                     targetVelocities=commands)
            
    def PostUpdate(self):
        self.follow_cam()
            
    def inverse_kinematics(self, v, theta):
        L = 0.28374 + 0.15751/2  # Distance from the center of the robot to each wheel
        r = 0.20074  # Radius of the wheels

        wl = (v - theta * L) / r
        wr = (v + theta * L) / r

        return wl, wr

if __name__ == "__main__":
     f = SelfBalanceSim()
