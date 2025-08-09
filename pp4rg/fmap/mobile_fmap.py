import numpy as np
import burg_toolkit as burg
from typing import Callable

from . import FeasibilityMapBase


class MobileFeasibilityMap:
    """
    Wrapper class for a feasibility map used to evaluate whether a mobile manipulator can feasibly reach a desired 
    end-effector (EE) pose, given its current base position.

    This class adapts a feasibility map built for a stationary manipulator to the mobile setting by transforming the
    query EE pose relative to the robot's base pose during map construction.

    Args:
        fmap (FeasibilityMapBase): FeasibilityMap constructed for stationary manipulator.
        collision_checker (Callable[[np.ndarray], bool]): A predicate function that takes a configuration (a NumPy array
            of shape (n,)) as input and returns True if the configuration is in collision with an obstacle, 
            and False if it is collision-free.
        forward_kinematics (Callable[[np.ndarray], np.ndarray]): Forward kinematics function of the mobile robotic arm. Should
            return np array: [x,y,z,h_x,h_y,h_z,h_w] where [h_x,h_y,h_z,h_w] is a quaternion.
        conf_to_base_tf (Callable[[np.ndarray], np.ndarray]): Function that maps a configuration of the mobile robot
            to a homogeneous transformation matrix representing its base pose.
        construction_base_tf (np.ndarray): Homogeneous transformation matrix representing the robot base pose 
            used during the construction of the feasibility map.
    """

    def __init__(self, fmap: FeasibilityMapBase, collision_checker: Callable[[np.ndarray], bool], forward_kinematics: Callable[[np.ndarray], np.ndarray],
                 conf_to_base_tf: Callable[[np.ndarray], np.ndarray], construction_base_tf: np.ndarray):
        self.fmap = fmap
        self.collision_checker = collision_checker
        self.forward_kinematics = forward_kinematics
        self.conf_to_base_tf = conf_to_base_tf
        self.construction_base_tf = np.array(construction_base_tf)  # ensure np.array

    def is_feasible(self, tf_ee: np.ndarray, conf: np.ndarray):
        """
        Returns true if target EE pose is feasibly reachable from robot's current base position. Base position is computed 
        from the provided robot configuration (using conf_to_base_tf).

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end effector in the world frame.
            conf (np.ndarray): Arbitrary configuration of the mobile manipulator that results in desired base position.
        """
        base_pose_tf = self.conf_to_base_tf(conf)
        # transformation to fmap's construction frame
        fmap_transform = self.construction_base_tf @ np.linalg.inv(base_pose_tf)

        def fmap_collision_checker(arm_conf):
            # adding correct XY position for collision check in the full configuration space
            return self.collision_checker(np.hstack((conf[:2], arm_conf)))

        def fmap_fk(arm_conf):
            full_conf = np.hstack((conf[:2], arm_conf))
            pos_quat = self.forward_kinematics(full_conf)
            ee_tf = burg.util.tf_from_pos_quat(pos_quat[:3], pos_quat[3:], convention='pybullet')
            fmap_ee_tf = fmap_transform @ ee_tf
            fmap_ee_pos, fmap_ee_quat = burg.util.position_and_quaternion_from_tf(fmap_ee_tf, convention='pybullet')
            return np.hstack((fmap_ee_pos, fmap_ee_quat))

        query_ee_tf = fmap_transform @ tf_ee
        return self.fmap.is_feasible(query_ee_tf, fmap_collision_checker, fmap_fk)

    def get_feasible_configurations(self, tf_ee: np.ndarray, conf: np.ndarray):
        """
        Returns arm configurations from the feasibility map that can reach the desired EE pose from 
        the robot's current base position.

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end effector in the world frame.
            conf (np.ndarray): Arbitrary configuration of the mobile manipulator that results in desired base position.
        """
        base_pose_tf = self.conf_to_base_tf(conf)
        # transformation to fmap's construction frame
        fmap_transform = self.construction_base_tf @ np.linalg.inv(base_pose_tf)

        def _fmap_collision_checker(arm_conf):
            # adding correct XY position for collision check in the full configuration space
            return self.collision_checker(np.hstack((conf[:2], arm_conf)))

        def _fmap_fk(arm_conf):
            full_conf = np.hstack((conf[:2], arm_conf))
            pos_quat = self.forward_kinematics(full_conf)
            ee_tf = burg.util.tf_from_pos_quat(pos_quat[:3], pos_quat[3:], convention='pybullet')
            fmap_ee_tf = fmap_transform @ ee_tf
            fmap_ee_pos, fmap_ee_quat = burg.util.position_and_quaternion_from_tf(fmap_ee_tf, convention='pybullet')
            return np.hstack((fmap_ee_pos, fmap_ee_quat))

        query_ee_tf = fmap_transform @ tf_ee
        arm_confs = self.fmap.get_feasible_configurations(query_ee_tf, _fmap_collision_checker, _fmap_fk)
        return arm_confs
