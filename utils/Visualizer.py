import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import numpy as np
import time
from os.path import abspath, dirname, join

class Visualizer:
    def __init__(self, path_to_urdf, path_to_robot_pkg=None):
        self.path_to_urdf = path_to_urdf
        if path_to_robot_pkg is None:
            path_to_robot_pkg = join(dirname(self.path_to_urdf), '../..')
        self.path_to_robot_pkg = abspath(path_to_robot_pkg)
        self.robot = RobotWrapper.BuildFromURDF(path_to_urdf, path_to_robot_pkg, pin.JointModelFreeFlyer())
        #self.robot = RobotWrapper.BuildFromURDF(path_to_urdf, path_to_robot_pkg)
        self.viewer = MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.camera_tf = meshcat.transformations.translation_matrix([0.8, -2.0, 0.2]) 
        self.zoom = 1.0
        self.play_speed = 1.0

    def display_meshcat(self, dt, q_traj, open=True):
        self.robot.setVisualizer(self.viewer)
        self.robot.initViewer(open=open)
        self.robot.loadViewerModel(rootNodeName='TrajectoryViewer')
        self.viewer.viewer["/Cameras/default"].set_transform(self.camera_tf)
        self.viewer.viewer["/Cameras/default/rotated/<object>"].set_property("zoom", self.zoom)
        self.viewer.viewer["/Background"].set_property("visible", True)
        self.viewer.viewer["/Background"].set_property("top_color", [0.9, 0.9, 0.9])
        self.viewer.viewer["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])

        sleep_time = dt / self.play_speed
        for q in q_traj:
            self.robot.display(q)
            time.sleep(sleep_time)