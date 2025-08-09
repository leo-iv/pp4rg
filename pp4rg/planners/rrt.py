"""
Single-query RRT planners for path planning in configuration space.
"""

from typing import Tuple, Optional
import time
import numpy as np

from ..tree import Tree
from ..spaces import ConfigurationSpace
from .utils import get_result_path, expand_rrt, expand_greedy_rrt


def rrt(cs: ConfigurationSpace, start: np.ndarray, goal: np.ndarray, time_limit=30, step_size=0.1, goal_bias=0.1,
        goal_tolerance=1e-5) -> Tuple[Optional[np.ndarray], Tree]:
    """
    Runs the RRT algorithm in the given configuration space. Uses straight-line interpolation with fixed step size for node expansion.

    Args:
        cs (ConfigurationSpace): The configuration space in which planning is performed.
        start (np.ndarray): The start configuration (shape: (DoF,)).
        goal (np.ndarray): The goal configuration (shape: (DoF,)).
        time_limit (float): Maximum allowed computation time in seconds. Defaults to 30.
        step_size (float): Step size for expansion. Defaults to 0.1.
        goal_bias (float): Probability of sampling the goal instead of a random configuration. Defaults to 0.1.
        goal_tolerance (float): Tolerance from the goal configuration. Defaults to 1e-5.

    Returns:
        Tuple (plan, tree):
            - plan (np.ndarray): Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
            - tree (Tree): The RRT search tree.
    """
    tree = Tree(cs.metric_space)
    tree.add_node(start)

    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        if np.random.random() < goal_bias:
            rand_conf = goal  # explore toward goal
        else:
            rand_conf = cs.get_random_conf()

        nearest_node = tree.get_nearest(rand_conf)
        new_conf = expand_rrt(nearest_node.coords, rand_conf, step_size)
        if not cs.line_collides(nearest_node.coords, new_conf):
            new_node = tree.add_node(new_conf, nearest_node)
            if cs.dist(new_conf, goal) < goal_tolerance:
                # goal configuration successfully connected to the tree (within tolerance)
                return get_result_path(new_node), tree

    return None, tree


def greedy_rrt(cs: ConfigurationSpace, start: np.ndarray, goal: np.ndarray, time_limit=30, step_size=0.1, goal_bias=0.1,
               goal_tolerance=1e-5) -> Tuple[Optional[np.ndarray], Tree]:
    """
    Runs the Greedy-RRT algorithm, which expands the tree greedily along a straight-line path toward sampled configurations.

    Args:
        cs (ConfigurationSpace): The configuration space in which planning is performed.
        start (np.ndarray): The start configuration (shape: (DoF,)).
        goal (np.ndarray): The goal configuration (shape: (DoF,)).
        time_limit (float): Maximum allowed computation time in seconds. Defaults to 30.
        step_size (float): Distance between successive nodes during expansion. Defaults to 0.1.
        goal_bias (float): Probability of sampling the goal instead of a random configuration. Defaults to 0.1.
        goal_tolerance (float): Tolerance from the goal configuration. Defaults to 1e-5.

    Returns:
        Tuple (plan, tree):
            - plan (np.ndarray): Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
            - tree (Tree): The RRT search tree.
    """
    tree = Tree(cs.metric_space)
    tree.add_node(start)

    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        if np.random.random() < goal_bias:
            rand_conf = goal  # explore toward goal
        else:
            rand_conf = cs.get_random_conf()

        nearest_node = tree.get_nearest(rand_conf)

        new_confs = expand_greedy_rrt(nearest_node.coords, rand_conf, step_size)
        for new_conf in new_confs:
            if cs.line_collides(nearest_node.coords, new_conf):
                break

            new_node = tree.add_node(new_conf, nearest_node)
            if cs.dist(new_conf, goal) < goal_tolerance:
                # goal configuration successfully connected to the tree (within tolerance)
                return get_result_path(new_node), tree

            nearest_node = new_node

    return None, tree
