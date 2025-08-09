import numpy as np
from typing import Tuple, Optional, Callable
import time
import torch

from ..spaces import ConfigurationSpace, MetricSpace, TaskSpace
from ..tree import Tree, Node
from .model import Autoencoder
from .dataset import normalize, denormalize
from ..planners.utils import expand_rrt, expand_j_plus, get_result_path_from_tree_confs


def latent_j_plus_rrt(cs: ConfigurationSpace, ts: TaskSpace, model: Autoencoder, self_collision_checker: Callable[[np.ndarray], bool], init_conf: np.ndarray,
                      goal_states: np.ndarray, time_limit=30, ls_step_size=0.1, ts_step_size=0.01, goal_bias=0.1, goal_tolerance=0.05) -> Tuple[Optional[np.ndarray], Tree]:
    """
    Runs the modified J+-RRT algorithm in the latent space created by the autoencoder network.

    Args:
        cs (ConfigurationSpace): The configuration space of the robot.
        ts (TaskSpace): Task space representation. Used to determine goal configuration (a configuration close to a grasping pose).
        model (Autoencoder): Autoencoder model used to encode configurations to latent space.
        self_collision_checker (Callable[[np.ndarray], bool]): A function that takes a configuration 
            (a numpy array of shape (n,)) and returns True if the configuration is in self-collision, 
            and False otherwise.
        init_conf (np.ndarray): The start configuration (shape: (DoF,)).
        goal_states (np.ndarray): The goal states (grasps) specified in the task space (shape: (number of goal states, 7)).
        time_limit (float): Maximum allowed computation time in seconds. Defaults to 30.
        ls_step_size (float): Step size for expansion in the latent space. Defaults to 0.1.
        ts_step_size (float): Step size for expansion in the task space (using the jacobian pseudoinverse). Defaults to 0.01.
        goal_bias (float): Probability of the goal extension. Defaults to 0.1.
        goal_tolerance (float): Tolerance from the goal configuration. Defaults to 1e-5.

    Returns:
        Tuple (plan, tree):
            - plan (np.ndarray): Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
            - tree (Tree): The latent RRT search tree.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    def encode(x_np: np.ndarray) -> np.ndarray:
        x_norm = normalize(x_np, cs)
        x_torch = torch.tensor(x_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            z_torch = model.encode(x_torch)
        return z_torch.cpu().numpy()

    def decode(z_np: np.ndarray) -> np.ndarray:
        z_torch = torch.tensor(z_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            x_torch = model.decode(z_torch)
        x_np = x_torch.cpu().numpy()
        return denormalize(x_np, cs)

    def random_self_collision_free_conf():
        q_rand = cs.get_random_conf()
        while self_collision_checker(q_rand):
            q_rand = cs.get_random_conf()
        return q_rand

    ms = MetricSpace([1] * model.latent_dim, [1] * model.latent_dim)
    latent_tree = Tree(ms)  # RRT in latent space
    z_init = encode(init_conf)
    start_node = latent_tree.add_node(z_init, conf=init_conf)

    task_tree = Tree(ts.metric_space)  # for nearest neighbor search in task space
    task_tree.add_node(ts.forward_kinematics(init_conf), conf=start_node)

    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        # random extension in latent space
        q_rand = random_self_collision_free_conf()
        z_rand = encode(q_rand)
        nearest_node = latent_tree.get_nearest(z_rand)
        z_new = expand_rrt(nearest_node.coords, z_rand, ls_step_size)
        q_new = decode(z_new)
        if not cs.line_collides(nearest_node.conf, q_new):
            new_node = latent_tree.add_node(z_new, parent=nearest_node, conf=q_new)
            task_tree.add_node(ts.forward_kinematics(q_new), conf=new_node)

        if np.random.random() < goal_bias:
            # explore toward a task space grasp with jacobian pseudo inverse
            goal_state = goal_states[np.random.randint(goal_states.shape[0])]  # random grasp
            nearest_ts_node = task_tree.get_nearest(goal_state)
            nearest_ls_node = nearest_ts_node.conf
            while True:
                q_new = expand_j_plus(nearest_ts_node.coords, nearest_ls_node.conf, goal_state, ts_step_size, ts)
                if (not cs.in_limits(q_new)) or cs.line_collides(nearest_ls_node.conf, q_new):
                    break
                z_new = encode(q_new)
                nearest_ls_node = latent_tree.add_node(z_new, parent=nearest_ls_node, conf=q_new)
                nearest_ts_node = task_tree.add_node(ts.forward_kinematics(q_new), conf=nearest_ls_node)
                if ts.dist(nearest_ts_node.coords, goal_state) < goal_tolerance:
                    return get_result_path_from_tree_confs(nearest_ls_node), latent_tree

    return None, latent_tree
