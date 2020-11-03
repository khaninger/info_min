# https://www.philipzucker.com/some-notes-on-drake-a-robotic-control-toolbox/
# https://drake.guzhaoyuan.com/introduction/drake-concept

#MESA_GL_VERSION_OVERRIDE=3.2 MESA_GLSL_VERSION_OVERRIDE=150 bazel run //tools:drake_visualizer #env variables are now in .bashrc. 
#bazel run examples/kuka_iiwa_arm/kuka_simulation

from pydrake.autodiffutils import AutoDiffXd
from pydrake.systems.framework import BasicVector, VectorSystem
import torch
import numpy as np
import copy

class SimpleContinuousTimeSystem(VectorSystem):
        def __init__(self):
                VectorSystem.__init__(self, 0, 3, False)
                self.DeclareContinuousState(2)
                self.design_params = {'b': 0.8, 'k_e': 1.0, 'x_wall': 1.0}
                self.design_params_te = {k : torch.tensor(v, requires_grad=True) for (k,v) in self.design_params.items()}
                self.x_te = [torch.tensor([0.0, 0.0])]
                self.prev_time_te = 0 # previous time, for the first-order approximation from 
                self.grads_calculated = False
                
        def DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
                u_ctrl = self.SimpleController(context, u, x)
                xdot[:] = self.Dynamics(context, u, x, u_ctrl, self.design_params)
                
                # Parallel computation w/ pytorch
                curr_time_te = context.get_time()
                dt = curr_time_te-self.prev_time_te
                self.prev_time_te = curr_time_te
                
                xdot_te = self.Dynamics(context, u, self.x_te[-1], u_ctrl, self.design_params_te)
                #xdot_te[0] = -self.x_te[-1][1]
                #xdot_te[1] = -self.design_params_te['b']*self.x_te[-1][1]-u_ctrl
                #if x[0] > self.design_params_te['x_wall']:
                #        xdot_te[1] = xdot_te[1]-self.design_params_te['k_e']*(self.design_params_te['x_wall']-self.x_te[-1][0])   
                x_new = self.x_te[-1] + dt*xdot_te
                self.x_te.append(x_new.clone().requires_grad_(True))
                if curr_time_te > 9.9 and not self.grads_calculated:
                        self.CalcGrad()
                        self.grads_calculated = True

        def Dynamics(self, context, u, x, u_ctrl, params):
                if isinstance(x, torch.Tensor):
                        xdot = torch.tensor((0.0, 0.0))
                else:
                        xdot = [0.0, 0.0]
                xdot[0] = -x[1]
                xdot[1] = -params['b']*x[1]-u_ctrl
                if x[0] > params['x_wall']:
                        xdot[1] = xdot[1]-params['k_e']*(params['x_wall']-x[0])
                return xdot 
                
        def DoCalcVectorOutput(self, context, u, x, y):
                y[0] = x[0] # Position
                y[1] = x[1] # Velocity
                y[2] = 0
                if x[0] > self.design_params['x_wall']:
                        y[2] = -self.design_params['k_e']*(self.design_params['x_wall']-x[0]) # Force
                        
        def CalcGrad(self):
                cost = torch.tensor(0.0)
                for x_curr in self.x_te:
                        if x_curr[0] > self.design_params_te['x_wall']:
                                f = -self.design_params_te['k_e']*(self.design_params_te['x_wall']-x_curr[0]) # Force
                                cost += torch.square(f)
                cost.backward()
                for (k, v) in self.design_params_te.items():
                        print('Grad of {} is: {}'.format(k, v.grad))
                        
        def SimpleController(self, context, u, x):
                if x[0] < self.design_params_te['x_wall']: # Contact force below threshold, positive force towards wall
                        u_ctrl = 0.5
                else:
                        u_ctrl = 1*x[1] # Add damping
                return u_ctrl
               

import matplotlib.pyplot as plt
from pydrake.all import *

# Create a simple block diagram containing our system.
builder = DiagramBuilder()
system = builder.AddSystem(SimpleContinuousTimeSystem())
#controller = builder.AddSystem(SimpleController())
logger = builder.AddSystem(SignalLogger(3))
#builder.Connect(system.get_output_port(0), controller.get_input_port(0))
#builder.Connect(controller.get_output_port(0), system.get_input_port(0))
builder.Connect(system.get_output_port(0), logger.get_input_port(0))
diagram = builder.Build()

# Create the simulator.
simulator = Simulator(diagram)

# Set the initial conditions, x(0).
state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
state.SetFromVector([0.0, 0.0])

# Simulate for 10 seconds.
simulator.AdvanceTo(10)

# Plot the results.
plt.plot(logger.sample_times(), logger.data()[0].transpose())
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()
