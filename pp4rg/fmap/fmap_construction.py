# This is a modified version of the "rm4d/construction.py" file from the RM4D library. See the README.md file for more details.
#
# RM4D is licensed under the MIT license.
# The original license text is retained in the LICENSE_RM4D file.

import numpy as np
from tqdm import tqdm
from burg_toolkit.util import tf_from_pos_quat
from abc import ABC, abstractmethod
from typing import Callable

from .fmap_base import FeasibilityMapBase


class FeasibilityMapConstructorBase(ABC):
    """
    Abstract base class for feasibility map constructors. These constructors use a more flexible,
    but less memory-efficient, data structure to support iterative construction.
    """

    @abstractmethod
    def get_indices_for_ee_pose(self, tf_ee: np.ndarray) -> tuple[int, ...]:
        """
        Computes map indices corresponding to a given end-effector pose.

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.

        Returns:
            tuple[int, ...]: Indices in the feasibility map corresponding to the pose.

        Raises:
            IndexError: If tf_ee is outside the bounds of the map.
        """
        pass

    @abstractmethod
    def add_conf(self, map_indices: tuple[int, ...], configuration: np.ndarray):
        """
        Stores a new robot configuration at the specified map location.

        Args:
            map_indices (tuple[int, ...]): Indices into the feasibility map.
            configuration (np.ndarray): Configuration vector to be stored (shape: (DoF)).
        """
        pass

    @abstractmethod
    def finalize_map(self) -> FeasibilityMapBase:
        """
        Finalizes and returns the constructed feasibility map.

        Returns:
            FeasibilityMapBase: The finalized feasibility map instance.
        """


class JointSpaceSampler:
    """
    Constructs a feasibility map by sampling random collision-free robot configurations
    and computing the corresponding end-effector pose using forward kinematics.

    Args:
        fmap (FeasibilityMapConstructorBase): Feasibility map to construct.
        joint_limits (np.array): Numpy array of shape (DoF, 2) with [min, max] limits for each joint.
        forward_kinematics (Callable): Function that maps a configuration to a task-space pose (a NumPy array 
            [x, y, z, h_x, h_y, h_z, h_w], with the last four elements forming a unit quaternion).
        collision_checker (Callable): Function that returns True if a given configuration of the robotic arm (shape: (DoF,)) 
            is in collision with obstacles.
    """

    def __init__(self, map: FeasibilityMapConstructorBase, joint_limits, forward_kinematics: Callable[[np.ndarray], np.ndarray],
                 collision_checker: Callable[[np.ndarray], bool] = None):
        self.map = map
        self.joint_limits = joint_limits
        self.forward_kinematics = forward_kinematics
        self.collision_checker = collision_checker
        if self.collision_checker is None:
            self.collision_checker = lambda c: False

    def _sample_random_conf(self):
        return np.random.uniform(low=self.joint_limits[:, 0], high=self.joint_limits[:, 1], size=(self.joint_limits.shape[0],))

    def _sample_conf(self, prevent_collision=True):
        rand_conf = self._sample_random_conf()

        if prevent_collision:
            while self.collision_checker(rand_conf):
                rand_conf = self._sample_random_conf()

        return rand_conf

    def sample(self, n_samples: int = 10000, prevent_collisions: bool = True):
        """
        Samples random robot configurations and adds them into the feasibility map.

        Args:
            n_samples (int): Number of configurations to sample. Defaults to 10,000.
            prevent_collisions (bool): If True, only collision-free configurations are added to the map. Defaults to True.
        """
        for _ in tqdm(range(n_samples), desc="Sampling Feasibility Map"):
            # sample random collision-free config
            rand_conf = self._sample_conf(prevent_collision=prevent_collisions)
            # get EE pose
            ee_pos_quat = self.forward_kinematics(rand_conf)
            tf_ee = tf_from_pos_quat(ee_pos_quat[:3], ee_pos_quat[3:], convention='pybullet')

            idcs = self.map.get_indices_for_ee_pose(tf_ee)
            self.map.add_conf(idcs, rand_conf)
