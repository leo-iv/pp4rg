from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class FeasibilityMapBase(ABC):
    """
    Abstract base class for the Feasibility map.
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
    def get_configurations(self, map_indices: tuple[int, ...]) -> np.ndarray:
        """
        Retrieves the configurations stored in the map at the specified indices. 
        Depending on the map type, these configurations may be incomplete (i.e., missing some DoF).

        Args:
            map_indices (tuple[int, ...]): Indices into the feasibility map.

        Returns:
            np.ndarray: Array of stored robot configurations or None if no configurations are stored at the given indices.
        """
        pass

    @abstractmethod
    def get_full_configurations(self, tf_ee: np.ndarray, forward_kinematics: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Retrieves full robot configurations corresponding to the given end-effector pose.

        In contrast to `get_configurations`, this method reconstructs the complete configuration vector,
        including any joints omitted during storage (e.g., base and wrist joints).

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.
            forward_kinematics (Callable): Function that maps a configuration to a task-space pose (a NumPy array 
                [x, y, z, h_x, h_y, h_z, h_w], with the last four elements forming a unit quaternion).

        Returns:
            np.ndarray: Array of stored robot configurations (shape: (n, DoF)).
        """
        pass

    def is_feasible(self, tf_ee: np.ndarray, collision_checker: Callable[[np.ndarray], bool],
                    forward_kinematics: Callable[[np.ndarray], np.ndarray]) -> bool:
        """
        Determines whether the given end-effector pose is feasibly reachable (i.e., reachable by a collision-free configuration).

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.
            collision_checker (Callable): Function that returns True if a given configuration of the robotic arm (shape: (DoF,)) 
                is in collision with obstacles.
            forward_kinematics (Callable): Function that maps a configuration to a task-space pose (a NumPy array 
                [x, y, z, h_x, h_y, h_z, h_w], with the last four elements forming a unit quaternion).

        Returns:
            bool: True if the pose is reachable by at least one collision-free configuration, False otherwise.
        """
        confs = self.get_full_configurations(tf_ee, forward_kinematics)
        return any(not collision_checker(conf) for conf in confs)

    def get_feasible_configurations(self, tf_ee: np.ndarray, collision_checker: Callable[[np.ndarray], bool],
                                    forward_kinematics: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Retrieves all collision-free configurations associated with the given end-effector pose.

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.
            collision_checker (Callable): Function that returns True if a given configuration of the robotic arm (shape: (DoF,))
                is in collision.
            forward_kinematics (Callable): Function that maps a configuration to a task-space pose (a NumPy array 
                [x, y, z, h_x, h_y, h_z, h_w], with the last four elements forming a unit quaternion).

        Returns:
            np.ndarray: Array of feasible configurations (shape: (n, DoF)).
        """
        confs = self.get_full_configurations(tf_ee, forward_kinematics)
        feasible_confs = [conf for conf in confs if not collision_checker(conf)]
        return np.array(feasible_confs)
