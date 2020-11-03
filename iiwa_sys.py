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

from info_optimizer import *

class iiwa_sys():
    def __init__(self, dt = 5e-4, N = 150, params = None, trj_decay = 0.7, x_w_cov = 1e-10):
        builder = DiagramBuilder()
        self.plant = builder.AddSystem(MultibodyPlant(time_step=dt))
        parser = Parser(self.plant)
        self.iiwa, hinge, self.bushing = self.add_models(parser)
        self.plant.Finalize()
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        self.simulator = Simulator(diagram, context)

        self.plant_context = self.plant.GetMyMutableContextFromRoot(context)
        nq = self.plant.num_positions()
        nv = self.plant.num_velocities()
        self.n_x = nq + nv
        self.n_u = self.plant.num_actuators()
        self.n_y = nq
        
        self.N = N
        self.dt = dt
        self.decay = trj_decay
        self.V = 1e-3*np.ones(self.n_y)
        self.W = np.concatenate((1e-6*np.zeros(nq), np.ones(nv)))
        self.W0 = np.concatenate((np.array([0., 0., 0., 0., 0., 0., 0., x_w_cov]),  1e-8*np.ones(nv)))
        
        q0 = np.array([-3, -.37, 1.5, -2.7, -1.0, 2.5, 1.5, 0.6])
        self.x0 = np.concatenate((q0, np.zeros(nv)))
        
        self.phi = {}
        #if params is not None:
        #    self.params_sym = np.array([sym.Variable(list(params.keys())[i]) for i in range(len(params))])
        #    par_array = np.array([self.params_sym[0], self.params_sym[1], self.params_sym[2]])
        #    bushing.SetForceStiffnessConstants(self.plant_context, par_array) 
            
        null_force = BasicVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.plant.GetInputPort("applied_generalized_force").FixValue(self.plant_context, null_force)
        self.plant.get_actuation_input_port().FixValue(self.plant_context, [0., 0., 0., 0., 0., 0., 0.])
        self.plant.SetPositions(self.plant_context, q0)
        #self.plant.SetVelocities(null_force)
        
        self.deriv = derivatives(self)
        
        self.x_trj, self.u_trj = None, None
        self.rollout()

    def add_models(self, parser):
        iiwa = parser.AddModelFromFile("/home/hanikevi/drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision_no_grav.sdf")
        #iiwa = parser.AddModelFromFile("/home/hanikevi/drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
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
        cupboard_door_spring = self.plant.AddForceElement(RevoluteSpring_[float](self.plant.GetJointByName("right_door_hinge"), nominal_angle = -0.4, stiffness = 1))
        bushing = LinearBushingRollPitchYaw_[float](   
            self.plant.GetFrameByName("iiwa_link_7"), 
            self.plant.GetFrameByName("handle"),
            [25, 25, 25], # Torque stiffness
            [1, 1, 1], # Torque damping
            [1100, 1100, 1100], # Linear stiffness
            [10, 10, 10], # Linear damping
        )
        bushing_element = self.plant.AddForceElement(bushing)
        
        return iiwa, hinge, bushing

    def cost_stage(self, x, u):
        #print(' u {}'.format(np.sum(0.01*u**2)))
        #print(' x {}'.format(15.0*(x[7]-0.0)**2))
        #print(' x {}'.format(np.sum(1e-4*x[8:]**2)))
        return np.sum(1e-4*u**2) + 1.0*(x[7]-2.0)**2 + np.sum(1e-4*x[8:]**2)

    def cost_final(self, x):
        return 1.0*(x[7]-2.0)**2 + np.sum(1e-4*x[8:]**2)

    def get_deriv(self, x, u):
        self.plant.SetPositionsAndVelocities(self.plant_context, x)
        lin = FirstOrderTaylorApproximation(self.plant, self.plant_context, self.plant.get_actuation_input_port().get_index(), self.plant.get_state_output_port().get_index())
        #print('A {}'.format(lin.A()))
        #print('B {}'.format(lin.B()))
        #print('C {}'.format(lin.C()[:self.n_y,:]))
        
        return lin.A(), lin.B(), lin.C()[:self.n_y, :]

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

    def dyn(self, x, u, phi = None, noise = False):
        x_next = self.plant.AllocateDiscreteVariables()
        aug_force = np.concatenate((u, np.array((0.0,))))
        self.plant.GetInputPort("applied_generalized_force").FixValue(self.plant_context, aug_force)
        
        self.plant.SetPositionsAndVelocities(self.plant_context, x)
        self.plant.CalcDiscreteVariableUpdates(self.plant_context, x_next)
        x_new = x_next.get_mutable_vector().get_value()
        if noise:
            noise = np.multiply(np.sqrt(self.W), np.random.randn(self.n_x))
            x_new = x_new + noise
        #print('x {}'.format(x_new))
        return x_new

    def obs(self, x, mode = None, noise = False, phi = None):
        y = self.plant.GetPositions(self.plant_context)
        #print(self.bushing.CalcBushingSpatialForceOnFrameA(self.plant_context).translational())
        if noise:
            y += np.multiply(np.sqrt(self.V), np.random.randn(self.n_y))
        #print('y {}'.format(y))
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
       
if __name__== "__main__":    
    params = OrderedDict()
    params['k1'] = 100
    params['k2'] = 100
    params['k3'] = 100
    params['k4'] = 100
    params['k5'] = 100
    params['k6'] = 100


    sys = iiwa_sys(params = params, trj_decay = 0.9)
    iLQR_ctrl = iLQR(sys, min_regu = 1e-11, state_regu = 1e-4)
    inf_opt = info_optimizer(iLQR_ctrl)
    iLQR_ctrl.run(do_plots = True, do_final_plot = True, do_fancy_plot = True, regu = .01, expected_cost_redu_thresh = 0.01)
    di, perf  = inf_opt.performance()
