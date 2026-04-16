# Self Balancing Robot  
### *End Term Project – Learning Based Control*

## Introduction  
This project focuses on the design, simulation, and control of a **self-balancing robot** using **learning-based control techniques**. 
---

## Installation  

Follow the steps below to set up the project environment.

### 1. Install PyBullet  
PyBullet is used for physics simulation of the robot.

```bash
pip install pybullet
```
## 2. Working with the code.
i. Import the class SelfBalanceSim from selfbalancebot.py and then create a class inheriting that. 
ii. Write overrides for methods like Start(), Update(), PreUpdate(), PostUpdate().
iii. The method *self.get_states()* will return the states $[\theta, \dot{\theta}, x, \dot{x}]$.
iv. Using *self.apply_input([wheel_l_vel, wheel_r_vel], mode = 'velocity' or 'torque')*, actuate the wheels.

## 3. (Simple) Example to apply some arbitrary input
```python
from selfbalancebot import SelfBalanceSim   # Importing the class
import pybullet as pb   # needed incase to call functions from the library
import numpy as np   # for linear algebra

class SelfBalancePID(SelfBalanceSim):
    '''
    Self Balance Bot implementation using a simple PID controller.
    '''
    # Update() will be called in every Frame in the loop
    def Update(self):
        # Make left wheel rotate by 10 rads/s and right wheel by -10 rads/s
        self.apply_input([10,-10], 'velocity')

# Need to write the following for it to work.
if __name__ == "__main__":
     f = SelfBalancePID()
```
## 4. (Simple) Example to display the states
```python
from selfbalancebot import SelfBalanceSim   # Importing the class
import pybullet as pb   # needed incase to call functions from the library
import numpy as np   # for linear algebra

class SelfBalancePID(SelfBalanceSim):
    '''
    Self Balance Bot implementation using a simple PID controller.
    '''
    # Update() will be called in every Frame in the loop
    def Update(self):
        # Make left wheel rotate by 10 rads/s and right wheel by -10 rads/s
        states = self.get_states()
        print(f"States: {states}")

        # After running self.get_states(), the same values will also be stored in self.states.
        print(f"self.states : {self.states}") # Same result as above.

# Need to write the following for it to work.
if __name__ == "__main__":
     f = SelfBalancePID()

```
## 5. (Advanced) Example PID controller
```python
from selfbalancebot import SelfBalanceSim   # Importing the class
import pybullet as pb   # needed incase to call functions from the library
import numpy as np   # for linear algebra

class SelfBalancePID(SelfBalanceSim):
    '''
    Self Balance Bot implementation using a simple PID controller.
    '''
    # Start() will be called once before begining the loop. Configure or declare variables here.
    def Start(self, start_pos=[0,0,0.5], start_orientation=pb.getQuaternionFromEuler([0,0,0]), model_path=r"/urdf/self_balance_bot.urdf"):
        super().Start(start_pos, start_orientation, model_path)  # Must write this

        # Declaring the PD gains
        self.Kp = [100, 1]
        self.Kd = [1, 10]

    # Will be called every frame in the loop
    def Update(self):
        # Call this to get the states or store them in self.states variable.
        self.get_states()
        # Seperate out the states from the array. self.states = [value of theta, value of theta_dot, value of x, value of x_dot] 
        theta, theta_dot, x, x_dot = self.states
        # The above is same as writing theta=self.states[0]; theta_dot=self.states[1]; x=self.states[2]; x_dot; self.states[3];

        # Use the state info to compute the control input using PD law. Here error =  theta -  theta_desired(=0) = theta. error_dot = theta_dot, as theta_desired is constant.
        wheel_vel = self.Kp[0]*theta - self.Kd[0]*theta_dot
        # Apply the input to both wheels
        self.apply_input([wheel_vel, wheel_vel], cmd_type='velocity')

    # This will be called at the end of each Frame in the Loop, after Physics is applied. Usefull to display or print stuffs for debugging
    def PostUpdate(self):
        # Print states for debugging
        print(f"States: {self.states}")

# Need to write the following for it to work.
if __name__ == "__main__":
     f = SelfBalancePID()
```
## 6. You may try to experiment with writing different sets of codes

