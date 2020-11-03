# https://www.philipzucker.com/some-notes-on-drake-a-robotic-control-toolbox/
# https://drake.guzhaoyuan.com/introduction/drake-concept

#MESA_GL_VERSION_OVERRIDE=3.2 MESA_GLSL_VERSION_OVERRIDE=150 bazel run //tools:drake_visualizer #env variables are now in .bashrc. 
#bazel run examples/kuka_iiwa_arm/kuka_simulation

from pydrake.autodiffutils import AutoDiffXd
from pydrake.systems.framework import BasicVector, VectorSystem
import torch
import numpy as np
import copy

class TwoMassSystem(VectorSystem):
        def __init__(self, design_params, ctrl):
                VectorSystem.__init__(self, 
                0, # Input:
                6, # Outputs: state and forces
                False) # No discrete state
                self.DeclareContinuousState(2, 2, 0)
                self.design_params = design_params
                self.design_params_te = {k : torch.tensor(v, requires_grad=True) for (k,v) in self.design_params.items()}
                self.x_te = [torch.tensor([0.0, 0.5, 0.0, 0.5])]
                self.ctrl = ctrl
                self.prev_time_te = 0 # previous time, for the first-order approximation from 
                self.grads_calculated = False
                self.cost = 0.0
                
        def DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
                xdot[:] = self.Dynamics(context, u, x, self.design_params)
                
                # Parallel computation w/ pytorch
                curr_time_te = context.get_time()
                dt = curr_time_te-self.prev_time_te
                self.prev_time_te = curr_time_te
                
                xdot_te = self.Dynamics(context, u, self.x_te[-1], self.design_params_te)
                x_new = self.x_te[-1] + dt*xdot_te
                self.x_te.append(x_new.clone().requires_grad_(True))
                
                if curr_time_te > 4.9 and not self.grads_calculated:
                        self.CalcGrad()
                        self.grads_calculated = True

        def ForceCalc(self, x, params):
                noise = np.random.normal([0,0], [1, 1], 2)    
                if isinstance(x, torch.Tensor):
                        f = torch.tensor([0.0, 0.0])
                        noise = torch.tensor(noise)
                else:
                        f = [0.0, 0.0]
                f[0] = params['k1']*(x[0]-x[2]) # Positive for compression
                if x[2] > params['x_wall']:
                        f[1] = params['k2']*(x[2]-params['x_wall']) # Force
                f_noise = f+params['force_noise_cov']*noise # Reparameterization trick ;)
        
                return f, f_noise

        def Dynamics(self, context, u, x, params):
                if self.ctrl is 'force':  u_ctrl = self.ForceController(context, u, x, params)
                elif self.ctrl is 'vel':  u_ctrl = self.VelController(context, u, x, params)
                else: u_ctrl = self.OpenLoopController(context, u, x)
                                         
                if isinstance(x, torch.Tensor):
                        xdot = torch.tensor((0.0, 0.0, 0.0, 0.0))
                        u_ctrl = torch.clamp(u_ctrl, -self.design_params['u_lim'], self.design_params['u_lim'])
                else:
                        xdot = [0.0, 0.0, 0.0, 0.0]
                        u_ctrl = np.clip(u_ctrl, -self.design_params['u_lim'],self.design_params['u_lim']) 
                xdot[0] = x[1]
                xdot[1] = (-params['b1']*x[1]+params['k1']*(x[2]-x[0])+u_ctrl)/params['m1']
                xdot[2] = x[3]
                xdot[3] = (-params['b2']*(x[3]-x[1])+params['k1']*(x[0]-x[2]))/params['m2']
                if x[2] > params['x_wall']:
                        xdot[3] += params['k2']*(params['x_wall']-x[2])/params['m2']
                return xdot 
                
        def DoCalcVectorOutput(self, context, u, x, y):
                y[0:4] = x[:] # Position, velocity
                _, y[4:6] = self.ForceCalc(x, self.design_params)

        def CalcGrad(self):
                cost = torch.tensor(0.0)
                for x_curr in self.x_te:
                        f = torch.tensor([0.0, 0.0])
                        if x_curr[2] > self.design_params_te['x_wall']:
                                f, _ = self.ForceCalc(x_curr, self.design_params_te)
                        cost += (f[1]-0.5)**2
                print('Thats-a gonn-a cost-a ya {}'.format(cost.item()))
                if cost.item():
                        cost.backward()
                        for (k, v) in self.design_params_te.items():
                                print('Grad of {} is: {}'.format(k, v.grad))
                self.cost = cost.item()
                        
        def OpenLoopController(self, context, u, x):
                u_ctrl = 0.5
                return u_ctrl
                
        def VelController(self, context, u, x, params):
                u_ctrl = params['vel_kp']*(params['vel_ref']-x[1])
                
                _, f_noise = self.ForceCalc(x,params)
                if f_noise[0] > 0.1:
                        u_ctrl = -params['u_lim']*x[1]**2/x[1] + 0.5
                        
                return u_ctrl
               
        def ForceController(self, context, u, x, params):                
                force_ref = params['vel_kp']*(params['vel_ref'] - x[1])
                
                _, f_noise = self.ForceCalc(x,params)
                
                # Simple PD force controller
                err = force_ref - f_noise[0]
                err_dot = params['k1']*(x[3]-x[1])
                u_ctrl = params['kp']*err + params['kd']*err_dot+f_noise[0]
                return u_ctrl

class ContactEstimator(VectorSystem):
        def __init__(self, design_params):
                VectorSystem.__init__(self, 
                6, # Input: All the two mass system states and the noisy forces
                3, # Outputs: state and forces
                False)
                
                self.cont_est = [0.0]
                self.mode_est = [0.5, 0.5]
                 
                sample_rate = 1/500 # time between estimate updates, in seconds
                self.DeclarePeriodicDiscreteUpdateEvent(sample_rate, 0.0, DoEstimate)
                                
        def DoEstimate(self, context, u, x):
                
                self.cont_est = self.cont_est
                
                self.mode_est = self.mode_est
                
        def ForceCalc(self, x, params):
                noise = np.random.normal([0,0], [1, 1], 2)    
                if isinstance(x, torch.Tensor):
                        f = torch.tensor((0.0, 0.0))
                        noise = torch.tensor(noise)
                else:
                        f = [0.0, 0.0]
                f[0] = params['k1']*(x[0]-x[2]) # Positive for compression
                if x[2] > params['x_wall']:
                        f[1] = params['k2']*(x[2]-params['x_wall']) # Force

                f_noise = f+params['force_noise_cov']*noise # Reparameterization trick ;)
        
                return f, f_noise

        def Dynamics(self, context, u, x, params):
                #u_ctrl = self.SimpleController(context, u, x)
                u_ctrl = self.ForceController(context, u, x, params)
                if isinstance(x, torch.Tensor):
                        xdot = torch.tensor((0.0, 0.0, 0.0, 0.0))
                else:
                        xdot = [0.0, 0.0, 0.0, 0.0]
                xdot[0] = x[1]
                xdot[1] = (-params['b1']*x[1]+params['k1']*(x[2]-x[0])+u_ctrl)/params['m1']
                xdot[2] = x[3]
                xdot[3] = (-params['b2']*(x[3]-x[1])+params['k1']*(x[0]-x[2]))/params['m2']
                if x[2] > params['x_wall']:
                        xdot[3] += params['k2']*(params['x_wall']-x[2])/params['m2']
                return xdot 
                
        def DoCalcVectorOutput(self, context, u, x, y):
                y[:] = x[:] # Should be all the states!



import matplotlib.pyplot as plt
from pydrake.all import *

design_params={ 'm1':0.25, 'b1':1.025, 'k1':25.0,# System parameters 
                'm2':0.15, 'b2':0.7, 'k2':50.0, 
                'x_wall':0.05,                   # Wall position
                'kp':3.5, 'kd':0.05,             # Torque control gains
                'vel_ref':0.1, 'vel_kp':5.0,     # Effectively the B_imp impedance gain, as well.
                'u_lim': 5.0, 'force_noise_cov':0.01
                }
k2s = (1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 100.0, 150.0, 250.0, 500.0)

ctrl = 'force'
perf_force = []
for k2_exp in k2s:
        design_params['k2'] = k2_exp
        # Create a simple block diagram containing our system.
        builder = DiagramBuilder()
        system = builder.AddSystem(TwoMassSystem(design_params, ctrl))
        logger = builder.AddSystem(SignalLogger(6))
        builder.Connect(system.get_output_port(0), logger.get_input_port(0))
        diagram = builder.Build()

        # Create the simulator.
        simulator = Simulator(diagram)

        # Set the initial conditions, x(0).
        state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
        state.SetFromVector([0.0, 0.05, 0.0, 0.05])

        simulator.get_mutable_integrator().set_fixed_step_mode(True)
        simulator.get_mutable_integrator().set_maximum_step_size(0.01)

        simulator.AdvanceTo(5)

        perf_force.append(system.cost)

ctrl = 'vel'
perf_vel = []
for k2_exp in k2s:
        design_params['k2'] = k2_exp
        # Create a simple block diagram containing our system.
        builder = DiagramBuilder()
        system = builder.AddSystem(TwoMassSystem(design_params, ctrl))
        logger = builder.AddSystem(SignalLogger(6))
        builder.Connect(system.get_output_port(0), logger.get_input_port(0))
        diagram = builder.Build()

        # Create the simulator.
        simulator = Simulator(diagram)

        # Set the initial conditions, x(0).
        state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
        state.SetFromVector([0.0, 0.05, 0.0, 0.05])

        simulator.get_mutable_integrator().set_fixed_step_mode(True)
        simulator.get_mutable_integrator().set_maximum_step_size(0.01)

        simulator.AdvanceTo(5)

        perf_vel.append(system.cost)

plt.plot(np.log(k2s), np.log(perf_vel))
plt.plot(np.log(k2s), np.log(perf_force))
plt.legend(('Velocity','Force'))
plt.show()
'''
# Plot the results.
plt.plot(logger.sample_times(), logger.data()[0].transpose(),'b')
plt.plot(logger.sample_times(), logger.data()[2].transpose(),'r')
plt.plot(logger.sample_times(), logger.data()[4].transpose(),'k')
plt.plot(logger.sample_times(), logger.data()[5].transpose(),'g')
plt.plot(logger.sample_times(), logger.data()[1].transpose())
plt.plot(logger.sample_times(), logger.data()[3].transpose())


plt.xlabel('t')
plt.ylabel('x(t), f(t)')
plt.legend(('x1', 'x2', 'f1', 'f2', 'v1', 'v2'))
plt.show()
''' 
