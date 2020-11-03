# start the meshcat server: ./bazel-bin/external/meshcat_python/meshcat-server
zmq_url='tcp://127.0.0.1:6000'

# mostly from https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/robot.ipynb#scrollTo=Qd245P5kY666

import numpy as np
import pydot

from pydrake.all import (
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, DiagramBuilder, 
    FindResourceOrThrow, GenerateHtml, InverseDynamicsController, 
    MultibodyPlant, Parser, Simulator, BasicVector, BasicVector_, 
    AutoDiffXd, initializeAutoDiff, autoDiffToGradientMatrix)
from pydrake.all import ( 
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, 
    DiagramBuilder_, RigidTransform, RotationMatrix, RollPitchYaw, Box,    
    CoulombFriction, FindResourceOrThrow, FixedOffsetFrame, 
    GeometryInstance, MeshcatContactVisualizer, Parser,  
    JointIndex, Simulator, FirstOrderTaylorApproximation,
    plot_system_graphviz)
from pydrake.multibody.tree import (RevoluteJoint_, SpatialInertia_, 
    UnitInertia_, RevoluteSpring_, LinearBushingRollPitchYaw_, 
    MultibodyForces_)

import pydrake.symbolic as sym
import matplotlib.pyplot as plt

from iiwa_sys import *
def add_models(parser, plant):
        iiwa = parser.AddModelFromFile("/home/hanikevi/drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision_no_grav.sdf")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
        
        box = Box(10., 10., 10.)
        X_WBox = RigidTransform([0, 0, -5])
        mu = 0.6
        plant.RegisterCollisionGeometry(plant.world_body(), X_WBox, box, "ground", CoulombFriction(mu, mu))
        plant.RegisterVisualGeometry(plant.world_body(), X_WBox, box, "ground", [.9, .9, .9, 1.0])
        planar_joint_frame = plant.AddFrame(FixedOffsetFrame("planar_joint_frame", plant.world_frame(), RigidTransform(RotationMatrix.MakeXRotation(np.pi/2))))

        X_WCylinder = RigidTransform([-0.75, 0, 0.5])
        hinge = parser.AddModelFromFile("/home/hanikevi/drake/examples/manipulation_station/models/simple_hinge.sdf")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"), X_WCylinder)
        cupboard_door_spring = plant.AddForceElement(RevoluteSpring_[float](plant.GetJointByName("right_door_hinge"), nominal_angle = -0.4, stiffness = 5))
        bushing = LinearBushingRollPitchYaw_[float](   
            plant.GetFrameByName("iiwa_link_7"), 
            plant.GetFrameByName("handle"),
            [10, 10, 10], # Torque stiffness
            [1, 1, 1], # Torque damping
            [100, 100, 100], # Linear stiffness
            [2, 2, 2], # Linear damping
        )
        bushing_element = plant.AddForceElement(bushing)
        
        return iiwa, hinge, bushing    

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=5e-4)
parser = Parser(plant, scene_graph)

iiwa, hinge, bushing = add_models(parser, plant)

plant.Finalize()
#plant_ad = plant.ToAutoDiffXd()
#plant_context_ad = plant_ad.CreateDefaultContext()
#plant_ad.RegisterAsSourceForSceneGraph(scene_graph)
#plant_ad.Finalize()
#builder.Connect(scene_graph.get_query_output_port(), plant_ad.get_geometry_query_input_port())

# Adds the MeshcatVisualizer and wires it to the SceneGraph.
meshcat = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
diagram = builder.Build()
context = diagram.CreateDefaultContext()
meshcat.load()
diagram.Publish(context)

nq = plant.num_positions()
nv = plant.num_velocities()
nx = nq + nv
nu = plant.num_actuators()
ort(iiwa)

plant.GetInputPort("applied_generalized_force").FixValue(plant_context, BasicVector([0., 0., 0., 0., 0., 0., 0.,0.]))
plant.get_actuation_input_port().FixValue(plant_context, [1.0, 0, 1.0, 0, 0, 0, 0])
plant.SetPositions(plant_context, [-3, -.37, 1.5, -2.7, -1.0, 2.5, 1.5, 0.6])

simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)

print(plant.EvalTimeDerivatives(plant_context).get_generalized_velocity())
print(plant.get_generalized_acceleration_output_port().Eval(plant_context))









#AUTODIFF
# Good ref: https://colab.research.google.com/github/RussTedrake/underactuated/blob/master/exercises/simple_legs/compass_gait_limit_cycle/compass_gait_limit_cycle.ipynb#scrollTo=vD3k3eCqi4LZ
#print(plant_ad.EvalTimeDerivatives(plant_context_ad).get_generalized_velocity())
#print(plant_ad.get_generalized_acceleration_output_port().Eval(plant_context_ad))


#print(plant.get_body_poses_output_port().get_index())
#approx = FirstOrderTaylorApproximation(plant, plant_context, 
#    plant.get_actuation_input_port(iiwa).get_index(),
#    plant.get_generalized_acceleration_output_port(iiwa).get_index())
#print(approx) 

#meshcat.start_recording()

simulator.AdvanceTo(10)

#meshcat.stop_recording()
#meshcat.publish_recording()
#print(context)

# good examples here
# https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/clutter.ipynb#scrollTo=XLAsUaA1JlGC


'''SCRAPS
# Adding a simple hinge
valve_arm = Box(0.25, 0.05, 0.05)
X_Wvalvearm = RigidTransform([0.5,0,0])
model_instance = parser.AddModelInstance("valve_arm_model")
plant.RegisterCollisionGeometry(plant.world_body(), X_Wvalvearm, valve_arm, "valve_arm", CoulombFriction(mu, mu))
plant.RegisterVisualGeometry(plant.world_body(), X_Wvalvearm, valve_arm, "valve_arm", [.9, 0.0, .9, 1.0])
#spatial_inertia = SpatialInertia_[float]()
spatial_inertia = SpatialInertia_[float]( mass = 1, p_PScm_E = [0,0,0], G_SP_E=UnitInertia_[float](1,1,1))
valve_arm_body = plant.AddRigidBody('valve_arm', model_instance, spatial_inertia)
valve_joint = plant.AddJoint(RevoluteJoint_[float](name="valve_joint", frame_on_parent = plant.world_frame(), frame_on_child = valve_arm_body.body_frame(), axis = [0, 0, 1], damping = 0.1))
valve_spring = plant.AddForceElement(RevoluteSpring_[float](joint = valve_joint, nominal_angle = 0.1, stiffness = 100.))
'''

# Panda
#parser.AddModelFromFile(FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf"))
#plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))

# Gripper - just had a lot of states and not 
#parser.AddModelFromFile(FindResourceOrThrow("drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50.sdf"))
#X_WGrip = RigidTransform(RollPitchYaw([3.1415179/2, 0.0, 3.1415179/2]), p = [0,0,0.114])
#plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName('body'), X_WGrip) 

# Gripper - just had a lot of states and not 
#parser.AddModelFromFile("/home/hanikevi/drake/examples/simple_gripper/simple_gripper.sdf")
#X_WGrip = RigidTransform(RollPitchYaw([3.1415179/2, 0.0, 3.1415179/2]), p = [0,0,0.114])
#plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName('body'), X_WGrip) 
