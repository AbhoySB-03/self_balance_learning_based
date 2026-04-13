'''
Contains definition of the PyBulletSimulation class, which provides a simple framework for running a physics simulation using the PyBullet library. The class handles the
Imports pybullet, pybullet_data, time modules.
Authors:
    - [Abhoy Sagar Bhowmik] <eez258580@iitd.ac.in>
    - [Om Prakash Behera] < >
    - [Palak Sharma] < >
    - [Prajwal Sharma] < >
'''
import pybullet as pb
import pybullet_data
import time

class PyBulletSimulation:
    def __init__(self, del_t=1/240): 
        '''
        PyBulletSimulation
        ===
        
        Class for creating and running a PyBullet physics simulation. It initializes 
        the simulation environment and runs simulation loop. This class can be inherited 
        and extended to create specific simulations by overriding the 
        Start, Update, PreUpdate, and PostUpdate methods.
        
        param del_t: Time step for the simulation (default is 1/240 seconds).
        '''
        self.physicsClient = pb.connect(pb.GUI)
        self.dt=del_t
        self.running=True
        self.gravity=(0.0,0.0,-9.8)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(*self.gravity)

        try:
            self.Start()
            while self.running:
                 if not pb.isConnected():
                     print("PyBullet connection lost. Stopping simulation.")
                     self.running = False
                     break
                 self.PreUpdate()
                 self.Update()
                 pb.stepSimulation()
                 self.PostUpdate()
                 time.sleep(self.dt)
        except KeyboardInterrupt:
            print("Simulation stopped by user.")
        finally:
            if pb.isConnected():
                pb.disconnect()                

    def Start(self):
        '''Start
        ===
        This method is called once at the beginning of simulation.
        Override this method to set up the simulation environment, 
        load models, and initialize variables.
        '''
        pass

    def Update(self):
        '''Update
        ===
        This method is called in every simulation step. 
        Override this method to implement the logic that should be executed 
        in each step of the simulation.'''
        pass

    def PreUpdate(self):
        '''PreUpdate
        ===
        This method is called just before Update(). 
        Override this method to implement any logic that 
        should be executed before the main update logic in each simulation step.'''
        pass

    def PostUpdate(self):
        '''PostUpdate
        ===
        This method is called after simulationStep which is called after Update(). 
        Override this method to implement any logic that 
        should be executed after the main update logic and simulation step.'''
        pass