# start the meshcat server: ./bazel-bin/external/meshcat_python/meshcat-server
zmq_url='tcp://127.0.0.1:6000'

# mostly from https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/robot.ipynb#scrollTo=Qd245P5kY666
# autodiff ref https://stackoverflow.com/questions/64565023/how-to-get-a-dynamic-which-we-can-be-applied-gradient-in-the-next-step-re-open


# autodiff tutorial: https://autodiff.github.io/tutorials/
# -maybe extend LinBush to accept symb or autodiff on force element? https://github.com/RobotLocomotion/drake/blob/e70e080e9d98075fd4691e98d3ae70d6322fc742/multibody/tree/linear_bushing_roll_pitch_yaw.h
# -or do a hand differentiation of linbush, then use the external forces thing...

# ref for doing autodiff on parameters: https://drake.mit.edu/pydrake/pydrake.multibody.plant.html
# relevant q for autodiff: https://stackoverflow.com/questions/62886835/approach-for-linearizing-nonlinear-system-around-decision-variables-x-u-from
import numpy as np
import pydot
from time import sleep
from copy import deepcopy
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
    plot_system_graphviz, PrintSimulatorStatistics, SignalLogger, 
    JacobianWrtVariable)
from pydrake.multibody.tree import (RevoluteJoint_, SpatialInertia_, 
    UnitInertia_, RevoluteSpring_, LinearBushingRollPitchYaw_, 
    MultibodyForces_)

import pydrake.symbolic as sym
import matplotlib.pyplot as plt
from collections import OrderedDict    

from info_optimizer import *

class iiwa_sys():
    def __init__(self, builder, dt = 5e-4, N = 150, params = None, trj_decay = 0.7, x_w_cov = 1e-5, door_angle_ref = 1.0, visualize = False):
        self.plant_derivs = MultibodyPlant(time_step=dt)
        parser = Parser(self.plant_derivs)
        self.derivs_iiwa, _, _ = self.add_models(self.plant_derivs, parser, params = params)
        self.plant_derivs.Finalize()
        self.plant_derivs_context = self.plant_derivs.CreateDefaultContext()
        self.plant_derivs.get_actuation_input_port().FixValue(self.plant_derivs_context, [0., 0., 0., 0., 0., 0., 0.])
        null_force = BasicVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.plant_derivs.GetInputPort("applied_generalized_force").FixValue(self.plant_derivs_context, null_force)

        self.plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step = dt)
        parser = Parser(self.plant, scene_graph)
        self.iiwa, self.hinge, self.bushing = self.add_models(self.plant, parser, params = params)
        self.plant.Finalize() # Finalize will assign ports for compatibility w/ the scene_graph; could be cause of the issue w/ first order taylor. 
        
        self.meshcat = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
        
        self.sym_derivs = False # If system should use symbolic derivatives; if false, autodiff
        self.custom_sim = False # If rollouts should be gathered with sys.dyn() calls
        
        nq = self.plant.num_positions()
        nv = self.plant.num_velocities()
        self.n_x = nq + nv
        self.n_u = self.plant.num_actuators()
        self.n_y = self.plant.get_state_output_port(self.iiwa).size()

        self.N = N
        self.dt = dt
        self.decay = trj_decay
        self.V = 1e-2*np.ones(self.n_y)
        self.W = np.concatenate((1e-7*np.ones(nq), 1e-4*np.ones(nv)))
        self.W0 = np.concatenate((1e-9*np.ones(nq),  1e-6*np.ones(nv)))
        self.x_w_cov = x_w_cov
        self.door_angle_ref = door_angle_ref
        
        self.q0 = np.array([-3.12, -0.17, 0.52, -3.11, 1.22, -0.75, -1.56, 0.55])
        #self.q0 = np.array([-3.12, -0.27, 0.52, -3.11, 1.22, -0.75, -1.56, 0.55])
        self.x0 = np.concatenate((self.q0, np.zeros(nv)))
        self.door_index = None
        
        self.phi = {}
        
    def ports_init(self, context):
        self.plant_context = self.plant.GetMyMutableContextFromRoot(context)    
        self.plant.SetPositionsAndVelocities(self.plant_context, self.x0)
        
        #door_angle = self.plant.GetPositionsFromArray(self.hinge, self.x0[:8])
        self.door_index = self.plant.GetJointByName('right_door_hinge').position_start()

        null_force = BasicVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.plant.GetInputPort("applied_generalized_force").FixValue(self.plant_context, null_force)
        self.plant.get_actuation_input_port().FixValue(self.plant_context, [0., 0., 0., 0., 0., 0., 0.])
        self.W0[self.door_index] = self.x_w_cov
        
    def add_models(self, plant, parser, params = None):
        iiwa = parser.AddModelFromFile("/home/hanikevi/drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision_no_grav.sdf")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
        
        #box = Box(10., 10., 10.)
        #X_WBox = RigidTransform([0, 0, -5])
        #mu = 0.6
        #plant.RegisterCollisionGeometry(plant.world_body(), X_WBox, box, "ground", CoulombFriction(mu, mu))
        #plant.RegisterVisualGeometry(plant.world_body(), X_WBox, box, "ground", [.9, .9, .9, 1.0])
        #planar_joint_frame = plant.AddFrame(FixedOffsetFrame("planar_joint_frame", plant.world_frame(), RigidTransform(RotationMatrix.MakeXRotation(np.pi/2))))

        X_WCylinder = RigidTransform([-0.75, 0, 0.5])
        hinge = parser.AddModelFromFile("/home/hanikevi/drake/examples/manipulation_station/models/simple_hinge.sdf")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"), X_WCylinder)
        #cupboard_door_spring = plant.AddForceElement(RevoluteSpring_[float](plant.GetJointByName("right_door_hinge"), nominal_angle = -0.4, stiffness = 10))
        if params is None:
                bushing = LinearBushingRollPitchYaw_[float](   
                    plant.GetFrameByName("iiwa_link_7"), 
                    plant.GetFrameByName("handle"),
                    [50, 50, 50], # Torque stiffness
                    [2., 2., 2.], # Torque damping
                    [5e4, 5e4, 5e4], # Linear stiffness
                    [80, 80, 80], # Linear damping
                )
        else:
                print('setting custom stiffnesses')
                bushing = LinearBushingRollPitchYaw_[float](   
                    plant.GetFrameByName("iiwa_link_7"), 
                    plant.GetFrameByName("handle"),
                    [params['k4'], params['k5'], params['k6']], # Torque stiffness
                    [2, 2, 2], # Torque damping
                    [params['k1'], params['k2'], params['k3']], # Linear stiffness
                    [100, 100, 100], # Linear damping
        )
        bushing_element = plant.AddForceElement(bushing)
        
        return iiwa, hinge, bushing

    def cost_stage(self, x, u):
        ctrl = 1e-5*np.sum(u**2)
        pos = 15.0*(x[self.door_index]-self.door_angle_ref)**2 
        vel = 1e-5*np.sum(x[8:]**2)
        return pos+ctrl+vel

    def cost_final(self, x):
        return 50*(1.0*(x[self.door_index]-self.door_angle_ref)**2 + np.sum(2.5e-4*x[8:]**2))

    def get_deriv(self, x, u):
        self.plant_derivs.SetPositionsAndVelocities(self.plant_derivs_context, x)
        lin = FirstOrderTaylorApproximation(self.plant_derivs, self.plant_derivs_context, self.plant.get_actuation_input_port().get_index(), self.plant.get_state_output_port(self.iiwa).get_index())        
        return lin.A(), lin.B(), lin.C()
        
    def get_param_deriv(self, x, u):
        # Using a closed-form solution; as currently DRAKE doesn't do support autodiff on parameters for LinearBusing.
        # Note dC is 0 - stiffness does not affect measurements of q, v
        W = self.plant.world_frame()
        I = self.plant.GetFrameByName("iiwa_link_7")
        H = self.plant.GetFrameByName("handle")
        self.plant.SetPositionsAndVelocities(self.plant_context, x)
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
        Jac_I = self.plant.CalcJacobianSpatialVelocity(self.plant_context, JacobianWrtVariable.kQDot, I, [0,0,0], W, W)        
        Jac_H = self.plant.CalcJacobianSpatialVelocity(self.plant_context, JacobianWrtVariable.kQDot, H, [0,0,0], W, W)
        dA = np.zeros((self.n_x**2, 3))
        dA_sq = np.zeros((self.n_x, self.n_x))
        for param_ind in range(3):
                JH = np.outer(Jac_H[param_ind,:], Jac_H[param_ind,:])
                JI = np.outer(Jac_I[param_ind,:], Jac_I[param_ind,:])
                dA_sq[8:, :8] = self.dt*np.linalg.inv(M).dot(JH+JI)
                dA[:,param_ind] = deepcopy(dA_sq.ravel())
                
       #print(np.sum(np.abs(dA), axis=0))
        return dA
                
    def reset(self):
        x_traj_new = np.zeros((self.N+1, self.n_x))
        x_traj_new[0,:] = self.x0 + np.multiply(np.sqrt(self.W0), np.random.randn(self.n_x))
        u_traj_new = np.zeros((self.N, self.n_u))
        #self.plant_context.SetDiscreteState(x_traj_new[0,:])
        self.plant.SetPositionsAndVelocities(self.plant_context, x_traj_new[0,:])
        return x_traj_new, u_traj_new

    def rollout(self):
        self.u_trj = np.random.randn(self.N, self.n_u)*0.001
        self.x_trj, _ = self.reset()
        for i in range(self.N):
            self.x_trj[i+1,:] = self.dyn(self.x_trj[i,:],self.u_trj[i], noise = True)   

    def dyn(self, x, u, phi = None, noise = False):
        x_next = self.plant.AllocateDiscreteVariables()
        self.plant.SetPositionsAndVelocities(self.plant_context, x) 
        self.plant.get_actuation_input_port(self.iiwa).FixValue(self.plant_context, u)
        self.plant.CalcDiscreteVariableUpdates(self.plant_context, x_next)
        #print(x_next.get_mutable_vector().get_value())
        x_new = x_next.get_mutable_vector().get_value()
        if noise:
            noise = np.multiply(np.sqrt(self.W), np.random.randn(self.n_x))
            x_new += noise       
        return x_new

    def obs(self, x, mode = None, noise = False, phi = None):
        y = self.plant.get_state_output_port(self.iiwa).Eval(self.plant_context)
        #print(self.bushing.CalcBushingSpatialForceOnFrameA(self.plant_context).translational())
        if noise:
            y += np.multiply(np.sqrt(self.V), np.random.randn(self.n_y))
        return y

    def cost(self, x_trj = None, u_trj = None):
        cost_trj = 0.0
        if x_trj is None:
            for i in range(self.N):
                cost_trj += self.cost_stage(self.x_trj[i,:], self.u_trj[i,:])
            cost_trj += self.cost_final(self.sys.plant, self.x_trj[-1,:])  
        else:
            for i in range(self.N):
                cost_trj += self.cost_stage(x_trj[i,:], u_trj[i,:])
            cost_trj += self.cost_final(x_trj[-1,:])  
        return cost_trj


if __name__== "__main__":
    params = OrderedDict()
    params['k4'] = 50
    params['k5'] = 50
    params['k6'] = 50
    # High stiff
    params['k1'] = 1e6
    params['k2'] = 1e6
    params['k3'] = 1e6


    grad1 = [-0.014, 0.243, 0.2673]
    grad2 = [0.005, 0.023, 0.069]
    grad = grad2
    # Optimized
    step = 3e6
    #params['k1'] -= step*grad[0]
    #params['k2'] -= step*grad[1]
    #params['k3'] -= step*grad[2]
    #params['k1'] = 1e4
    #params['k2'] = 1e4
    #params['k3'] = 1e4


    builder = DiagramBuilder()
    dt = 2.5e-4
    N = 500
    sys = iiwa_sys(builder, params = params, trj_decay = 0.5, dt = dt, N = N, x_w_cov = 1e-5, door_angle_ref = -0.5)
    iLQR_ctrl = iLQR(sys, min_regu = 5e-5, state_regu = 0.0)
    builder.AddSystem(iLQR_ctrl)
    builder.Connect(sys.plant.get_state_output_port(sys.iiwa), iLQR_ctrl.get_input_port(0))
    builder.Connect(iLQR_ctrl.get_output_port(0), sys.plant.get_actuation_input_port())

    logger = builder.AddSystem(SignalLogger(7))
    builder.Connect(iLQR_ctrl.get_output_port(), logger.get_input_port())

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    sys.ports_init(context)
    sys.deriv = derivatives(sys)
    sys.meshcat.load()
    diagram.Publish(context)
    
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(True)
    simulator.get_mutable_integrator().set_fixed_step_mode(True)
    simulator.get_mutable_integrator().set_maximum_step_size(dt)
    
    iLQR_ctrl.simulator = simulator
                
    inf_opt = info_optimizer(iLQR_ctrl)
    print('Tuning LQR')
    iLQR_ctrl.run(max_iter = 5, regu = 5e-5, expected_cost_redu_thresh = 1.0, do_final_plot = False)

    #print(inf_opt.grad_directed_info(0))
    #print(inf_opt.grad_directed_info(1))    
    #print(inf_opt.grad_directed_info(2))   
    
    print('Evaluating')
    inf_opt.plot_w_di()
    
    print('Recording video')
    # Let's take a video
    sys.meshcat.start_recording()
    iLQR_ctrl.forward_pass()
    iLQR_ctrl.forward_pass()
    iLQR_ctrl.forward_pass()    
    sys.meshcat.stop_recording()
    sys.meshcat.publish_recording()

