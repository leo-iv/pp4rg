import numpy as np
from numpy.linalg import norm
from typing import Callable, Optional

from . import mpnn


class MetricSpace:
    """
    Represents the topology of a configuration space manifold. The topology is defined using the same format 
    as in the MPNN library. For more information, see their paper: https://lavalle.pl/software/mpnn/mpnn.html

    Args:
        topology (list of int): An array specifying the topology of each dimension, using numbers from 1 to 3. 
            - 1 represents Euclidean space (R^1).
            - 2 represents circular space (S^1).
            - 3 represents real projective space (RP^3), used to model 3D rotations via unit quaternions. 
              There must always be four consecutive 3s to represent a valid quaternion.

            For example, the SE(3) space (the space of all translations and rotations of a rigid body) is 
            represented as [1, 1, 1, 3, 3, 3, 3].

        scale (list of float): A scale factor for each dimension used in metric calculations. Must be the same 
            length as the topology array.
    """

    def __init__(self, topology, scale):
        assert len(topology) == len(scale), "Topology and scale of the MetricSpace have to have same length."
        self.dim = len(topology)
        self.topology = np.array(topology, dtype=np.int32)
        self.scale = np.array(scale, dtype=np.float32)

    def dist(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """   
        Computes the distance between two points in the metric space.

        Args:
            point1 (np.ndarray): A NumPy array representing the first point in the metric space.
            point2 (np.ndarray): A NumPy array representing the second point in the metric space.

        Returns:
            float: The distance between the two points in the metric space.
        """
        return mpnn.dist(point1, point2, self.topology, self.scale)


class ConfigurationSpace:
    """
    Represents an abstract configuration space for a path planning problem. Encapsulates configuration space bounds 
    and a collision detection mechanism.

    Args:
        limits (np.ndarray): Limits of the configuration space. Numpy array of shape (n, 2) with [min, max] for each dimension.
        collision_checker (Callable[[np.ndarray], bool]): A predicate function that takes a configuration (a NumPy array 
            of shape (n,)) as input and returns True if the configuration is in collision with an obstacle, 
            and False if it is collision-free.
        collision_step_size (float): The resolution of collision checking. Lower values provide more accurate 
            checks but increase computational cost.
        metric_space (MetricSpace): A `MetricSpace` instance used for distance calculations. Defaults to Euclidean R^n space.
    """

    def __init__(self, limits: np.ndarray,
                 collision_checker: Callable[[np.ndarray], bool],
                 collision_step_size: float = 0.01,
                 metric_space: Optional[MetricSpace] = None):
        self.dim = limits.shape[0]  # dimension of the C-space
        self.limits = np.array(limits, dtype=np.float32)
        self.check_collision = collision_checker
        self.collision_step_size = collision_step_size
        self.metric_space = metric_space
        if self.metric_space is None:
            self.metric_space = MetricSpace([1 for _ in range(self.dim)], [1 for _ in range(self.dim)])

    def get_random_conf(self) -> np.ndarray:
        """
        Samples a random configuration uniformly from the configuration space limits.
        Note: This does not account for the topology defined in the MetricSpace.

        Returns:
            np.ndarray: A random configuration within the defined limits.
        """
        return np.random.uniform(low=self.limits[:, 0], high=self.limits[:, 1], size=(self.dim,))

    def in_limits(self, conf: np.ndarray) -> bool:
        """
        Checks whether a given configuration lies within the configuration space limits.

        Args:
            conf (np.ndarray): The configuration to check.

        Returns:
            bool: True if the configuration is within limits, False otherwise.
        """
        return np.all(self.limits[:, 0] <= conf) and np.all(conf <= self.limits[:, 1])

    def point_collides(self, conf: np.ndarray) -> bool:
        """
        Checks whether a given configuration collides with obstacles.

        Args:
            conf (np.ndarray): The configuration to check.

        Returns:
            bool: True if the configuration is in collision, False otherwise.
        """
        return self.check_collision(conf)

    def line_collides(self, conf_a: np.ndarray, conf_b: np.ndarray) -> bool:
        """
        Checks whether the straight-line path between two configurations collides with any obstacles (using the resolution specified in the constructor).

        Args:
            conf_a (np.ndarray): Start configuration.
            conf_b (np.ndarray): End configuration.

        Returns:
            bool: True if line collides with obstacles, False otherwise.
        """
        if np.allclose(conf_a, conf_b):
            return False

        diff = conf_b - conf_a
        dist = norm(diff)
        step_vect = (diff / dist) * self.collision_step_size
        conf = conf_a
        for i in range(int(dist // self.collision_step_size) + 1):
            if self.check_collision(conf):
                return True
            conf = conf + step_vect

        if self.check_collision(conf_b):
            return True

        return False

    def dist(self, conf1: np.ndarray, conf2: np.ndarray) -> float:
        """
        Computes the distance between two configurations using the specified metric.

        Args:
            conf1 (np.ndarray): First configuration.
            conf2 (np.ndarray): Second configuration.

        Returns:
            float: The distance between the two configurations.
        """
        return self.metric_space.dist(conf1, conf2)


class TaskSpace:
    """
    Represents the robot's task space. Currently, only SE(3) space is supported, where the task state 
    is represented as a NumPy array of shape (7,): [x, y, z, h_x, h_y, h_z, h_w], with the last four elements 
    forming a unit quaternion.

    The primary intended usage of this class is to represent TaskSpace for a mobile manipulator, but it should be possible to run some
    of the planners even for other types of robots.

    Args:
        limits (np.ndarray): Limits for the 3D translation part of the task space. Numpy array of shape (3, 2) with [min, max] for each dimension.
        forward_kinematics (Callable[[np.ndarray], np.ndarray]): Function that maps a configuration space vector (of shape `(DoF,)`)
            to a task-space vector of shape (7,) in SE(3).
        inverse_kinematics (Callable[[np.ndarray], np.ndarray]): Function that maps a task-space vector (shape (7,)) to a configuration space vector.
            Optional - only some planning algorithms require it.
        jacobian (Callable[[np.ndarray], np.ndarray]): Function that computes the Jacobian matrix of the forward kinematics for the given configuration.
            Assumed to be expressed in the robot's initial base frame (as in pybullet). Optional - only some planning algorithms require it.
        init_base_to_world_tf (np.ndarray): Homogenous transformation matrix which specifies the initial base position of the robot in the world frame.
        rot_scale (float): Scaling factor for the rotational part of the distance metric. Defaults to one.
    """

    def __init__(self, limits: np.ndarray,
                 forward_kinematics: Callable[[np.ndarray], np.ndarray],
                 inverse_kinematics: Callable[[np.ndarray], np.ndarray] = None,
                 jacobian: Callable[[np.ndarray], np.ndarray] = None,
                 init_base_to_world_tf: np.ndarray = None,
                 rot_scale: float = 1):
        assert limits.shape == (3, 2), "Incorrect limits for TaskSpace specified. Only SE(3) is currently supported."
        self.limits = limits
        self.forward_kinematics = forward_kinematics
        self.inverse_kinematics = inverse_kinematics
        self.jacobian = jacobian
        self.init_base_to_world_tf = init_base_to_world_tf if init_base_to_world_tf is not None else np.eye(4)
        self.world_to_init_base_tf = np.linalg.inv(self.init_base_to_world_tf)
        self.dim = 7
        self.metric_space = MetricSpace([1, 1, 1, 3, 3, 3, 3], [1, 1, 1, rot_scale, rot_scale, rot_scale, rot_scale])

    def get_random_state(self) -> np.ndarray:
        """
        Samples a random state uniformly from the task space.

        Returns:
            np.ndarray: A 7D task-space state [x, y, z, h_x, h_y, h_z, h_w], with a uniformly sampled unit quaternion.
        """
        # sample translation
        pos = np.random.uniform(low=self.limits[:, 0], high=self.limits[:, 1], size=(3,))
        # sample uniform quaternion (K. Shoemake, 1992)
        u = np.random.uniform(low=0, high=1, size=(3,))
        s1 = np.sqrt(1 - u[0])
        s2 = np.sqrt(u[0])
        orn = np.array([s1 * np.sin(2 * np.pi * u[1]), s1 * np.cos(2 * np.pi * u[1]),
                        s2 * np.sin(2 * np.pi * u[2]), s2 * np.cos(2 * np.pi * u[2])])
        return np.hstack((pos, orn))

    def in_limits(self, state: np.ndarray) -> bool:
        """
        Checks whether the given state is within the task space translation limits. (Does not check for unit quaternion).

        Args:
            state (np.ndarray): A 7D task-space state vector.

        Returns:
            bool: True if the translation part is within limits, False otherwise.
        """
        return np.all(self.limits[:, 0] <= state[:3]) and np.all(state[:3] <= self.limits[:, 1])

    def dist(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Computes the distance between two task-space states.

        Args:
            state1 (np.ndarray): First state vector (7D).
            state2 (np.ndarray): Second state vector (7D).

        Returns:
            float: The SE(3) distance between the two states.
        """
        return self.metric_space.dist(state1, state2)
