from typing import Tuple, Optional
import time
import numpy as np
import burg_toolkit as burg
from scipy.spatial.transform import Rotation as R

from ..tree import Tree
from ..spaces import ConfigurationSpace, TaskSpace
from .utils import get_result_path, expand_rrt, expand_j_plus


def j_plus_rrt(cs: ConfigurationSpace, ts: TaskSpace, start_conf: np.ndarray, goal_states: np.ndarray, time_limit=30,
               cs_step_size=0.1, ts_step_size=0.1, goal_bias=0.1, goal_tolerance=1e-5) -> Tuple[Optional[np.ndarray], Tree]:
    """
    Runs the J+-RRT algorithm for planning in grasping tasks. Based on: Vahrenkamp et al., "Humanoid Motion Planning for
    Dual-Arm Manipulation and Re-Grasping Tasks".

    Args:
        cs (ConfigurationSpace): The configuration space in which planning is performed.
        ts (TaskSpace): Task space representation, with forward_kinematics and jacobian methods.
        start_conf (np.ndarray): The start configuration (shape: (DoF,)).
        goal_states (np.ndarray): The goal states specified in the task space (shape: (number of goal states, 7)).
        time_limit (float): Maximum allowed computation time in seconds. Defaults to 30.
        cs_step_size (float): Step size for expansion in the configuration space. Defaults to 0.1.
        ts_step_size (float): Step size for expansion in the task space (using the jacobian pseudoinverse). Defaults to 0.1.
        goal_bias (float): Probability of the goal extension. Defaults to 0.1.
        goal_tolerance (float): Tolerance from the goal configuration. Defaults to 1e-5.

    Returns:
        Tuple (plan, tree):
            - plan (np.ndarray): Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
            - tree (Tree): The RRT search tree constructed in configuration space.
    """

    conf_tree = Tree(cs.metric_space)  # stores configurations for standard RRT expansion in CS
    start_node = conf_tree.add_node(start_conf)

    task_tree = Tree(ts.metric_space)  # for nearest neighbor search when connecting goal configurations
    task_tree.add_node(ts.forward_kinematics(start_conf), conf=start_node)

    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        # normal random RRT expand
        rand_conf = cs.get_random_conf()
        nearest_node = conf_tree.get_nearest(rand_conf)
        new_conf = expand_rrt(nearest_node.coords, rand_conf, cs_step_size)
        if not cs.line_collides(nearest_node.coords, new_conf):
            new_node = conf_tree.add_node(new_conf, nearest_node)
            task_tree.add_node(ts.forward_kinematics(new_conf), conf=new_node)

        if np.random.random() < goal_bias:
            # explore toward a task space grasp with jacobian pseudo inverse
            goal_state = goal_states[np.random.randint(goal_states.shape[0])]  # random grasp
            nearest_ts_node = task_tree.get_nearest(goal_state)
            nearest_cs_node = nearest_ts_node.conf
            while True:
                new_conf = expand_j_plus(nearest_ts_node.coords, nearest_cs_node.coords, goal_state, ts_step_size, ts)
                if (not cs.in_limits(new_conf)) or cs.line_collides(nearest_cs_node.coords, new_conf):
                    break
                nearest_cs_node = conf_tree.add_node(new_conf, nearest_cs_node)
                nearest_ts_node = task_tree.add_node(ts.forward_kinematics(new_conf), conf=nearest_cs_node)
                if ts.dist(nearest_ts_node.coords, goal_state) < goal_tolerance:
                    return get_result_path(nearest_cs_node), conf_tree

    return None, conf_tree


def j_plus_ik_rrt(cs: ConfigurationSpace, ts: TaskSpace, start_conf: np.ndarray, goal_states: np.ndarray,
                  ik_solutions: np.ndarray, time_limit=30, cs_step_size=0.1, ts_step_size=0.1, goal_bias=0.1,
                  goal_tolerance=1e-5) -> Tuple[Optional[np.ndarray], Tree]:
    """
    Runs a slightly modified version of the J+-RRT algorithm. This variant biases configuration space expansion 
    toward known IK solutions, while also guiding expansion in task space via the Jacobian pseudoinverse.

    Args:
        cs (ConfigurationSpace): The configuration space in which planning is performed.
        ts (TaskSpace): Task space representation, with forward_kinematics and jacobian methods.
        start_conf (np.ndarray): The start configuration (shape: (DoF,)).
        goal_states (np.ndarray): The goal states specified in the task space (shape: (number of goal states, 7)).
        ik_solutions (np.ndarray): Precomputed IK solutions for the task-space goals (shape: (N, DoF)). Each row is a 
        robot configuration toward which the search may be biased.
        time_limit (float): Maximum allowed computation time in seconds. Defaults to 30.
        cs_step_size (float): Step size for expansion in the configuration space. Defaults to 0.1.
        ts_step_size (float): Step size for expansion in the task space (using the jacobian pseudoinverse). Defaults to 0.1.
        goal_bias (float): Probability of sampling the goal instead of a random configuration. Defaults to 0.1.
        goal_tolerance (float): Tolerance from the goal configuration. Defaults to 1e-5.

    Returns:
        Tuple (plan, tree):
            - plan (np.ndarray): Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
            - tree (Tree): The RRT search tree constructed in configuration space.
    """
    conf_tree = Tree(cs.metric_space)  # stores configurations for standard RRT expansion in CS
    start_node = conf_tree.add_node(start_conf)

    task_tree = Tree(ts.metric_space)  # for nearest neighbor search when connecting goal configurations
    task_tree.add_node(ts.forward_kinematics(start_conf), conf=start_node)

    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        # normal RRT expand (with ik solution goal bias)
        if np.random.random() < goal_bias:
            rand_conf = ik_solutions[np.random.randint(ik_solutions.shape[0]), :]  # explore toward ik solution
        else:
            rand_conf = cs.get_random_conf()
        nearest_node = conf_tree.get_nearest(rand_conf)
        new_conf = expand_rrt(nearest_node.coords, rand_conf, cs_step_size)
        if not cs.line_collides(nearest_node.coords, new_conf):
            new_node = conf_tree.add_node(new_conf, nearest_node)
            task_tree.add_node(ts.forward_kinematics(new_conf), conf=new_node)

        if np.random.random() < goal_bias:
            # explore toward a task space grasp with jacobian pseudo inverse
            goal_state = goal_states[np.random.randint(goal_states.shape[0])]  # random grasp
            nearest_ts_node = task_tree.get_nearest(goal_state)
            nearest_cs_node = nearest_ts_node.conf
            while True:
                new_conf = expand_j_plus(nearest_ts_node.coords, nearest_cs_node.coords, goal_state, ts_step_size, ts)
                if (not cs.in_limits(new_conf)) or cs.line_collides(nearest_cs_node.coords, new_conf):
                    break
                nearest_cs_node = conf_tree.add_node(new_conf, nearest_cs_node)
                nearest_ts_node = task_tree.add_node(ts.forward_kinematics(new_conf), conf=nearest_cs_node)
                if ts.dist(nearest_ts_node.coords, goal_state) < goal_tolerance:
                    return get_result_path(nearest_cs_node), conf_tree

    return None, conf_tree
