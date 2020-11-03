# start the meshcat server: ./bazel-bin/external/meshcat_python/meshcat-server
zmq_url='tcp://127.0.0.1:6000'

# mostly from https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/robot.ipynb#scrollTo=Qd245P5kY666

import numpy as np
import pydot

from pydrake.all import (
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, DiagramBuilder, 
    FindResourceOrThrow, GenerateHtml, InverseDynamicsController, 
    MultibodyPlant, Parser, Simulator)
from pydrake.all import ( 
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, 
    DiagramBuilder, RigidTransform, RotationMatrix, RollPitchYaw, Box,    
    CoulombFriction, FindResourceOrThrow, FixedOffsetFrame, 
    GeometryInstance, MeshcatContactVisualizer, Parser,  
    JointIndex, Simulator
)
from pydrake.multibody.tree import RevoluteJoint_, SpatialInertia_, UnitInertia_, RevoluteSpring_, LinearBushingRollPitchYaw_

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)

parser = Parser(plant, scene_graph)

#parser.AddModelFromFile(FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"))
parser.AddModelFromFile("/home/hanikevi/drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision_no_grav.sdf")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

box = Box(10., 10., 10.)
X_WBox = RigidTransform([0, 0, -5])
mu = 0.6
plant.RegisterCollisionGeometry(plant.world_body(), X_WBox, box, "ground", CoulombFriction(mu, mu))
plant.RegisterVisualGeometry(plant.world_body(), X_WBox, box, "ground", [.9, .9, .9, 1.0])
planar_joint_frame = plant.AddFrame(FixedOffsetFrame("planar_joint_frame", plant.world_frame(), RigidTransform(RotationMatrix.MakeXRotation(np.pi/2))))

X_WCylinder = RigidTransform([-0.75, 0, 0.5])
parser.AddModelFromFile("/home/hanikevi/drake/examples/manipulation_station/models/simple_hinge.sdf")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"), X_WCylinder)
#cupboard_door_spring = plant.AddForceElement(RevoluteSpring_[float](plant.GetJointByName("right_door_hinge"), nominal_angle = -0.4, stiffness = 1))

bushing = plant.AddForceElement(LinearBushingRollPitchYaw_[float](   
    plant.GetFrameByName("iiwa_link_7"), 
    plant.GetFrameByName("handle"),
    [25, 25, 25], # Torque stiffness
    [10, 10, 10], # Torque damping
    [1100, 1100, 1100], # Linear stiffness
    [125, 125, 125], # Linear damping
))

plant.Finalize()

# Adds the MeshcatVisualizer and wires it to the SceneGraph.
meshcat = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)

diagram = builder.Build()
context = diagram.CreateDefaultContext()

meshcat.load()
diagram.Publish(context)

print(context)

plant_context = plant.GetMyMutableContextFromRoot(context)
plant.SetPositions(plant_context, [-3, -.37, 1.5, -2.7, -1.0, 2.5, 1.5, 0.6])
plant.get_actuation_input_port().FixValue(plant_context, [10.0, 0, 10.0, 0, 0, 0, 0])


simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)

#meshcat.start_recording()
simulator.AdvanceTo(10)
#meshcat.stop_recording()
#meshcat.publish_recording()
print(context)

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
