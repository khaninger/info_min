# mostly from https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/robot.ipynb#scrollTo=Qd245P5kY666
# autodiff ref https://stackoverflow.com/questions/64565023/how-to-get-a-dynamic-which-we-can-be-applied-gradient-in-the-next-step-re-open


# autodiff tutorial: https://autodiff.github.io/tutorials/
# -maybe extend LinBush to accept symb or autodiff on force element? https://github.com/RobotLocomotion/drake/blob/e70e080e9d98075fd4691e98d3ae70d6322fc742/multibody/tree/linear_bushing_roll_pitch_yaw.h
# -or do a hand differentiation of linbush, then use the external forces thing...

# ref for doing autodiff on parameters: https://drake.mit.edu/pydrake/pydrake.multibody.plant.html
# relevant q for autodiff: https://stackoverflow.com/questions/62886835/approach-for-linearizing-nonlinear-system-around-decision-variables-x-u-from
import numpy as np
import pydot

from pydrake.all import (
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, DiagramBuilder, 
    FindResourceOrThrow, GenerateHtml, InverseDynamicsController, 
    MultibodyPlant, Parser, Simulator, BasicVector, BasicVector_, 
    AutoDiffXd, initializeAutoDiff, autoDiffToGradientMatrix, FirstOrderTaylorApproximation)
from pydrake.all import ( 
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, 
    DiagramBuilder_, RigidTransform, RotationMatrix, RollPitchYaw, Box,    
    CoulombFriction, FindResourceOrThrow, FixedOffsetFrame, 
    GeometryInstance, MeshcatContactVisualizer, Parser,  
    JointIndex, Simulator_, FirstOrderTaylorApproximation,
    plot_system_graphviz)
from pydrake.multibody.tree import (RevoluteJoint_, SpatialInertia_, 
    UnitInertia_, RevoluteSpring_, LinearBushingRollPitchYaw_, 
    MultibodyForces_)

import pydrake.symbolic as sym
import matplotlib.pyplot as plt
from collections import OrderedDict    

class iiwa_deriv():
    def __init__(self, dt = 5e-3, N = 150, params = None, trj_decay = 0.95):
        builder = DiagramBuilder()
        self.plant = builder.AddSystem(MultibodyPlant(time_step=dt))
        parser = Parser(self.plant)
        iiwa, hinge, bushing = self.add_models(parser)
        self.plant.Finalize()
        
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        self.plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        nq = plant.num_positions()
        nv = plant.num_velocities()
        self.n_x = nq + nv
        self.n_u = plant.num_actuators()
        self.n_y = self.nq
        
        self.N = N
        self.dt = dt
        self.decay = trj_decay
        self.V = 1e-3*np.ones(n_y)
        self.W = 1.0*np.array([1e-7, 1e-2, 1e-7, 4e-2, 0])
        self.W0 = np.array([3e-4, 3e-4, 3e-4, 3e-4, x_w_cov])
        
        q0 = np.array([-3, -.37, 1.5, -2.7, -1.0, 2.5, 1.5, 0.6])
        self.x0 = q0.append(np.zeros(nv))
        
        #if params is not None:
        #    self.params_sym = np.array([sym.Variable(list(params.keys())[i]) for i in range(len(params))])
        #    par_array = np.array([self.params_sym[0], self.params_sym[1], self.params_sym[2]])
        #    bushing.SetForceStiffnessConstants(self.plant_context, par_array) 
            
        null_force = BasicVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.plant.GetInputPort("applied_generalized_force").FixValue(self.plant_context, null_force)
        self.plant.get_actuation_input_port().FixValue(self.plant_context, [0., 0., 0., 0., 0., 0., 0.])
        self.plant.SetPositions(self.plant_context, q0)

        self.deriv = derivatives(self)
        
        self.x_trj, self.u_trj = None, None
        self.rollout()

    def add_models(self, parser):
        iiwa = parser.AddModelFromFile("/home/hanikevi/drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision_no_grav.sdf")
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("iiwa_link_0"))

        box = Box(10., 10., 10.)
        X_WBox = RigidTransform([0, 0, -5])
        mu = 0.6
        #plant.RegisterCollisionGeometry(plant.world_body(), X_WBox, box, "ground", CoulombFriction(mu, mu))
        #plant.RegisterVisualGeometry(plant.world_body(), X_WBox, box, "ground", [.9, .9, .9, 1.0])
        planar_joint_frame = self.plant.AddFrame(FixedOffsetFrame("planar_joint_frame", self.plant.world_frame(), RigidTransform(RotationMatrix.MakeXRotation(np.pi/2))))

        X_WCylinder = RigidTransform([-0.75, 0, 0.5])
        hinge = parser.AddModelFromFile("/home/hanikevi/drake/examples/manipulation_station/models/simple_hinge.sdf")
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("base"), X_WCylinder)
        #cupboard_door_spring = plant.AddForceElement(RevoluteSpring_[float](plant.GetJointByName("right_door_hinge"), nominal_angle = -0.4, stiffness = 1))
        bushing = LinearBushingRollPitchYaw_[float](   
            self.plant.GetFrameByName("iiwa_link_7"), 
            self.plant.GetFrameByName("handle"),
            [25, 25, 25], # Torque stiffness
            [10, 10, 10], # Torque damping
            [1100, 1100, 1100], # Linear stiffness
            [125, 125, 125], # Linear damping
        )
        bushing_element = self.plant.AddForceElement(bushing)
        return iiwa, hinge, bushing

    def cost_stage(self, x, u):
        return 0.1*u**2 + 5.0*x[7]**2 + 0.1*x[8:]**2

    def cost_final(self, x):
        return 5.0*x[7]**2 + 0.1*x[8:]**2

    def get_deriv(self, x, u):
        #self.plant_context.SetTimeStateAndParametersFrom(context)
        lin = FirstOrderTaylorApproximation(self.plant, self.plant_context, self.plant.get_actuation_input_port().get_index(), self.plant.GetPositions().get_index())
        return lin.A(), lin.B(), lin.C()

    def reset(self):
        x_traj_new = np.zeros((self.N+1, self.n_x))
        x_traj_new[0,:] = self.x0 + np.multiply(np.sqrt(self.W0), np.random.randn(self.n_x))
        u_traj_new = np.zeros((self.N, self.n_u))
        return x_traj_new, u_traj_new

    def rollout(self):
        self.u_trj = np.random.randn(self.N, self.n_u)*0.001
        self.x_trj, _ = self.reset()
        for i in range(self.N):
            self.x_trj[i+1,:] = self.dyn(self.x_trj[i,:],self.u_trj[i], noise = True)   

    def dyn(self, x, u, phi = None, mode = None, noise = False):
        # x = [position, velocity, position, velocity, wall_pos], u = [force], mode = free_space || contact
        m = sym if x.dtype == object else np # check for autodiff            
        dt = self.dt
        m1, b1 = 0.2, 9.8
        m2, b2 = 0.11, 5.6 

        if phi is None:
            k1 = self.phi['k1']
            k2 = self.phi['k2']
        else:
            k1 = phi[0]
            k2 = phi[1]

        if not mode:
            mode = self.mode_check(x)

        if mode == 'free_space':
            x_next = x+np.array([dt*x[1], dt/m1*(u[0]-b1*x[1]-k1*(x[0]-x[2])), dt*x[3], dt/m2*(-b2*x[3]+k1*(x[0]-x[2])), 0.0], dtype=object)
        elif mode == 'contact':
            x_next = x+np.array([dt*x[1], dt/m1*(u[0]-b1*x[1]-k1*(x[0]-x[2])), dt*x[3], dt/m2*(-b2*x[3]+k1*(x[0]-x[2])-k2*(x[2]-x[4])), 0.0], dtype=object)
            
        if noise:
            noise = np.multiply(np.sqrt(self.W), np.random.randn(self.n_x))
            x_next += noise
        return x_next 

    def mode_check(self,x):
        if x[2] > x[-1]:
            return 'contact'
        else:
            return 'free_space'

    def cost_stage(self, x, u):
        contact_force = self.smax(self.phi['k2']*(x[2]-x[-1]))
        c = 0.05*(3-contact_force)**2 + 0.05*(x[2]-x[-1])**2+ 0.1*x[1]**2 + 0.1*x[3]**2+1e-8*u**2
        
        c = 1/self.phi['k2']**2*(3-self.phi['k2']*(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2+1e-8*u**2

        #c = 0.3*(0.01-(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2+1e-8*u**2
        return c 

    def cost_final(self, x):
        contact_force = self.smax(self.phi['k2']*(x[2]-x[-1]))
        c = 0.05*(3-contact_force)**2 + 0.05*(x[2]-x[-1])**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        #c = 1/self.phi['k2']**2*(3-self.phi['k2']*(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        #c = 0.003*(1-self.phi['k2']*(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        #c = 0.3*(0.01-(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        return 1*c     

    def smax(self, x, beta = 1):
        m = sym if x is object else np # check for autodiff            
        return 0.5*(m.sqrt(x**2+beta**2)+x)
    
    def obs(self, x, mode = None, noise = False, phi = None):
        m = sym if x.dtype == object else np # check for autodiff 
        if phi is None:
            k1 = self.phi['k1']
            k2 = self.phi['k2']
        else:
            k1 = phi[0]
            k2 = phi[1]           
        if not mode:
            mode = self.mode_check(x)
        if mode == 'free_space': 
            y = [x[0], k1*(x[2]-x[0])]
        elif mode == 'contact':
            y = [x[0], k1*(x[2]-x[0])]
        if noise:
            y += np.multiply(np.sqrt(self.V), np.random.randn(2))
        return y

    def cost(self, x_trj = None, u_trj = None):
        cost_trj = 0.0
        if x_trj is None:
            for i in range(self.u_trj.shape[0]):
                cost_trj += self.cost_stage(self.x_trj[i,:], self.u_trj[i,:])
            cost_trj += self.cost_final(self.x_trj[-1,:])  
        else:
            for i in range(u_trj.shape[0]):
                cost_trj += self.cost_stage(x_trj[i,:], u_trj[i,:])
            cost_trj += self.cost_final(x_trj[-1,:])  
        return cost_trj

    def soft_update_trj(self, x_trj_new, u_trj_new, Sigma):
        self.x_trj = self.decay*self.x_trj + (1-self.decay)*x_trj_new
        self.u_trj = self.decay*self.u_trj + (1-self.decay)*u_trj_new
        
    def update_params(self, params):
   
params = OrderedDict()
params['k1'] = 100
params['k2'] = 100
params['k3'] = 100
params['k4'] = 100
params['k5'] = 100
params['k6'] = 100

der = iiwa_deriv(params = params)
A, B, C = der.stage([1.0], [1.0])
print(A)
