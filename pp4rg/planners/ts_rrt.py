from typing import Tuple, Optional
import time
import numpy as np
import burg_toolkit as burg
from scipy.spatial.transform import Rotation as R

from ..tree import Tree
from ..spaces import ConfigurationSpace, TaskSpace
from .utils import expand_j_plus, get_result_path_from_tree_confs

def ts_rrt(cs: ConfigurationSpace, ts: TaskSpace, start_conf: np.ndarray, goal_states: np.ndarray, time_limit=30,
           step_size=0.1, goal_bias=0.1, goal_tolerance=1e-5) -> Tuple[Optional[np.ndarray], Tree]:
    """
    Runs the TS-RRT algorithm for planning in grasping tasks.Uses the pseudoinverse of the Jacobian matrix for task-space expansion.
    Based on: Shkolnik and Tedrake, "Path Planning in 1000+ Dimensions Using a Task-Space Voronoi Bias".

    Args:
        cs (ConfigurationSpace): The configuration space in which planning is performed.
        ts (TaskSpace): Task space representation, with forward_kinematics and jacobian methods.
        start_conf (np.ndarray): The start configuration (shape: (DoF,)).
        goal_states (np.ndarray): The goal states specified in the task space (shape: (number of goal states, 7)).
        time_limit (float): Maximum allowed computation time in seconds. Defaults to 30.
        step_size (float): Step size for expansion in the task space (using the jacobian pseudoinverse). Defaults to 0.1.
        goal_bias (float): Probability of sampling the goal instead of a random configuration. Defaults to 0.1.
        goal_tolerance (float): Tolerance from the goal configuration. Defaults to 1e-5.

    Returns:
        Tuple (plan, tree):
            - plan (np.ndarray): Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
            - tree (Tree): The RRT search tree constructed in task space.
    """
    tree = Tree(ts.metric_space)
    tree.add_node(coords=ts.forward_kinematics(start_conf), parent=None, conf=start_conf)

    last_goal_state = goal_states[np.random.randint(goal_states.shape[0])]
    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        if np.random.random() < goal_bias:
            rand_state = goal_states[np.random.randint(goal_states.shape[0])]  # explore toward randomly selected goal
            last_goal_state = rand_state
        else:
            rand_state = ts.get_random_state()

        nearest_node = tree.get_nearest(rand_state)
        while True:
            new_conf = expand_j_plus(nearest_node.coords, nearest_node.conf, rand_state, step_size, ts)
            if (not cs.in_limits(new_conf)) or cs.line_collides(nearest_node.conf, new_conf):
                break
            new_state = ts.forward_kinematics(new_conf)
            new_node = tree.add_node(coords=new_state, parent=nearest_node, conf=new_conf)
            if ts.dist(new_state, last_goal_state) < goal_tolerance:
                # goal configuration successfully connected to the tree (within tolerance)
                return get_result_path_from_tree_confs(new_node), tree
            if ts.dist(new_state, rand_state) < goal_tolerance:
                # arrived at the random state successfully
                break
            nearest_node = new_node

    return None, tree
