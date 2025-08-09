import numpy as np
import burg_toolkit as burg
from scipy.spatial.transform import Rotation as R

from ..tree import Node


def get_result_path(goal_node: Node):
    # extract the result path from the tree by backtracking through parent pointers
    path = []
    node = goal_node

    while node.parent is not None:
        path.append(node.coords)
        node = node.parent
    path.append(node.coords)
    path.reverse()

    return np.array(path)

def get_result_path_from_tree_confs(goal_node):
    # extracts the result path from node.conf instead of node.coords
    # needed (for example) for the TS-RRT algorithm
    path = []
    node = goal_node

    while node.parent is not None:
        path.append(node.conf)
        node = node.parent
    path.append(node.conf)
    path.reverse()

    return np.array(path)


def expand_rrt(nearest_conf, rand_conf, step_size):
    # make a "straight-line" expansion in the direction of rand_conf by step_size
    dir = rand_conf - nearest_conf
    dist = np.linalg.norm(dir)
    if dist <= step_size:
        return rand_conf  # expand by rand_conf if closer to the tree than step size
    return nearest_conf + (dir * step_size / dist)  # move by step size in the rand_conf direction


def expand_greedy_rrt(nearest_conf, rand_conf, step_size):
    # make a "greedy straight-line" expansion in the direction of rand_conf
    dist = np.linalg.norm(rand_conf - nearest_conf)
    steps = int(dist // step_size) + 1  # new nodes are step_size apart
    interpolation = (nearest_conf*(1-i/steps) + (rand_conf*i/steps) for i in range(1, steps + 1))  # exclude first
    return interpolation


def _compute_rot_difference(tf1, tf2):
    r1 = tf1[:3, :3]  # extract rotation matrices
    r2 = tf2[:3, :3]
    r_rel = r2 @ r1.T
    rotvec = R.from_matrix(r_rel).as_rotvec()
    return rotvec


def _compute_pose_difference(x1, x2, world_to_init_base_tf):
    pos1, quat1 = x1[:3], x1[3:]
    pos2, quat2 = x2[:3], x2[3:]
    tf1 = burg.util.tf_from_pos_quat(pos1, quat1, convention='pybullet')
    tf2 = burg.util.tf_from_pos_quat(pos2, quat2, convention='pybullet')
    tf1 = world_to_init_base_tf @ tf1  # move to robots initial base frame (pybullet computes jacobians in this frame)
    tf2 = world_to_init_base_tf @ tf2
    pos_diff = tf2[:3, 3] - tf1[:3, 3]  # xyz position difference
    rotvec = _compute_rot_difference(tf1, tf2)  # rotation difference
    return np.hstack((pos_diff, rotvec))


def _limit_ts_step_size(difference_vector, max_step):
    norm = np.linalg.norm(difference_vector)
    if norm > max_step:
        difference_vector = difference_vector / norm * max_step
    return difference_vector


def expand_j_plus(nearest_pose, nearest_conf, new_pose, step_size, ts):
    # expansion of the rrt tree in the task space using the pseudoinverse of the jacobian matrix
    diff = _compute_pose_difference(nearest_pose, new_pose, ts.world_to_init_base_tf)
    diff = _limit_ts_step_size(diff, step_size)
    J = ts.jacobian(nearest_conf)
    J_plus = np.linalg.pinv(J)
    conf_diff = J_plus @ diff
    return nearest_conf + conf_diff
