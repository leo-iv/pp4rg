import numpy as np
import time
import burg_toolkit as burg
from typing import Optional

from ..spaces import ConfigurationSpace, TaskSpace
from ..tree import Tree
from ..fmap import MobileFeasibilityMap
from .j_plus_rrt import j_plus_rrt
from .rrt import greedy_rrt
from .utils import get_result_path, expand_greedy_rrt


def _precompute_grasps_states(grasps_tf):
    """
    Converts end-effector poses from homogeneous transformation matrices to [pos, quat] format.
    """
    grasp_states = np.empty((grasps_tf.shape[0], 7))
    for i, tf in enumerate(grasps_tf):
        pos, quat = burg.util.position_and_quaternion_from_tf(tf, convention='pybullet')
        grasp_states[i, :] = np.hstack((pos, quat))
    return grasp_states


def _get_full_plan(goal_node, init_conf):
    base_pose_path = get_result_path(goal_node)
    return np.hstack((base_pose_path, np.tile(init_conf[3:], (base_pose_path.shape[0], 1))))


def _is_suitable(base_pos, init_conf, grasps_tf, mobile_fmap: MobileFeasibilityMap):
    """
    Checks if a given base position allows the manipulator to reach any of the grasp targets.
    """
    full_conf = np.hstack((base_pos, init_conf[3:]))
    for grasp in grasps_tf:
        if mobile_fmap.is_feasible(grasp, full_conf):
            return True
    return False


def find_base_positions(cs: ConfigurationSpace, mobile_fmap: MobileFeasibilityMap, init_conf: np.ndarray, grasps: np.ndarray,
                        n_solutions: int, time_limit: float, secondary_time_limit: float, base_epsilon: float, step_size: float):
    """
    Finds suitable base positions for a mobile manipulator using a greedy RRT approach.

    The planner searches for base configurations from which the manipulator can reach one of the
    target end-effector poses  (checked using its feasibility map). The robot's arm remains in a fixed
    "low-volume" (extracted from init_conf) configuration during planning.

    Args:
        cs (ConfigurationSpace): Full configuration space of the mobile manipulator (base + arm). It is assumed that the
            first two coordinates correspond to the base position (x, y) and the third controls the base joint of the arm.
        mobile_fmap (MobileFeasibilityMap): Feasibility map built for the robotic arm.
        init_conf (np.ndarray): Initial full configuration of the robot (used to extract initial base and fixed "low-volume"
            arm configuration).
        grasps (np.ndarray): Goal end effector poses as homogeneous transformations - np.array of shape (N, 4, 4).
        max_solutions (int): Maximum number of feasible base paths to find.
        time_limit (float): Maximum total planning time in seconds.
        secondary_time_limit (float): Time allowed after the first solution is found to continue finding others.
        base_epsilon (float): Required distance between individual solutions (ending base positions).
        step_size (float): Step size for greedy RRT expansion in base space.

    Returns:
        (result_base_confs, result_plans):
        - result_base_confs (np.ndarray): Suitable base configurations including fixed arm part (shape: (N, DoF)).
        - result_plans (list[np.ndarray]): List of full paths for each base conf in result_base_confs.
    """
    def base_collision_checker(conf):
        full_conf = np.hstack((conf, init_conf[3:]))  # add "low volume" arm configuration
        return cs.check_collision(full_conf)

    base_cs = ConfigurationSpace(cs.limits[:3, :], base_collision_checker, cs.collision_step_size)

    result_base_confs = []
    result_plans = []

    # greedy-RRT algorithm
    tree = Tree(base_cs.metric_space)
    tree.add_node(init_conf[:3])

    first_solution_time = float('inf')
    enough_solutions = False
    start_time = time.time()
    while (
        (time.time() - start_time) < time_limit
        and (time.time() - first_solution_time) < secondary_time_limit
        and not enough_solutions
    ):
        rand_base_pose = base_cs.get_random_conf()
        nearest_node = tree.get_nearest(rand_base_pose)
        new_base_poses = expand_greedy_rrt(nearest_node.coords, rand_base_pose, step_size)
        for new_base_pose in new_base_poses:
            if base_cs.line_collides(nearest_node.coords, new_base_pose):
                break

            new_node = tree.add_node(new_base_pose, nearest_node)
            if (
                _is_suitable(new_base_pose, init_conf, grasps, mobile_fmap)
                and all(np.linalg.norm(new_base_pose[:2] - base_pose[:2]) >= base_epsilon for base_pose in result_base_confs)
            ):
                if len(result_base_confs) == 0:  # found first solution
                    first_solution_time = time.time()
                result_base_confs.append(np.hstack((new_base_pose, init_conf[3:])))
                result_plans.append(_get_full_plan(new_node, init_conf))
                if len(result_base_confs) >= n_solutions:
                    enough_solutions = True
                    break

            nearest_node = new_node

    return np.array(result_base_confs), result_plans


def plan_arm_motion_to_compact_conf(cs: ConfigurationSpace, init_conf: np.ndarray, compact_arm_conf: np.ndarray,
                                    time_limit: float, step_size: float):
    """
    Plans a collision-free motion for the arm of a mobile manipulator from init_arm_conf to compact_arm_conf using Greedy-RRT.

    Args:
        cs (ConfigurationSpace): Full configuration space of the mobile manipulator (base + arm). It is assumed that the first two 
            coordinates correspond to the base position (x, y) in XY plane.
        init_conf (np.ndarray): Initial configuration of the mobile manipulator.
        compact_arm_conf (np.ndarray): Compact configuration of the robotic arm.
        time_limit (float): Maximum allowed planning time in seconds.
        step_size (float): Step size for expansion in configuration space.

    Returns:
        np.ndarray: Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
    """
    def arm_collision_checker(arm_conf):
        full_conf = np.hstack((init_conf[:2], arm_conf))
        return cs.check_collision(full_conf)

    arm_cs = ConfigurationSpace(cs.limits[2:, :], arm_collision_checker, cs.collision_step_size)
    # big goal bias since we assume that the robot "starts" in obstacle free environment
    arm_plan, _ = greedy_rrt(arm_cs, init_conf[2:], compact_arm_conf, time_limit, step_size, 0.8, 0.05)
    if arm_plan is None:
        return None
    full_plan = np.hstack((np.tile(init_conf[:2], (arm_plan.shape[0], 1)), arm_plan))
    return full_plan


def plan_arm_motion(cs: ConfigurationSpace, ts: TaskSpace, init_conf: np.ndarray, grasp_states: np.ndarray, time_limit: float,
                    cs_step_size: float, ts_step_size: float, goal_bias: float, goal_tol: float):
    """
    Plans a collision-free motion for the arm of a mobile manipulator using J+-RRT, while keeping the base fixed.

    Args:
        cs (ConfigurationSpace): Full configuration space of the mobile manipulator (base + arm). It is assumed that the first two 
            coordinates correspond to the base position (x, y) in XY plane.
        ts (TaskSpace): Task space of the mobile manipulator.
        init_conf (np.ndarray): Initial configuration (with the desired fixed base position).
        grasp_states (np.ndarray): Goal end effector position's. np.array (N, 7) where each row is a task space state (position + quaternion).
        time_limit (float): Maximum allowed planning time in seconds.
        cs_step_size (float): Step size for expansion in configuration space.
        ts_step_size (float): Step size for expansion in task space.
        goal_bias (float): Probability of sampling the goal instead of a random configuration.
        goal_tol (float): Tolerance for reaching the goal.

    Returns:
        np.ndarray: Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
    """
    def arm_collision_checker(arm_conf):
        full_conf = np.hstack((init_conf[:2], arm_conf))
        return cs.check_collision(full_conf)

    def arm_fk(arm_conf):
        full_conf = np.hstack((init_conf[:2], arm_conf))
        return ts.forward_kinematics(full_conf)

    def arm_jacobian(arm_conf):
        full_conf = np.hstack((init_conf[:2], arm_conf))
        full_jacobian = ts.jacobian(full_conf)
        return full_jacobian[:, 2:]  # get rid of base motion

    arm_cs = ConfigurationSpace(cs.limits[2:, :], arm_collision_checker, cs.collision_step_size)
    arm_ts = TaskSpace(ts.limits, forward_kinematics=arm_fk, jacobian=arm_jacobian,
                       init_base_to_world_tf=ts.init_base_to_world_tf)

    arm_plan, _ = j_plus_rrt(arm_cs, arm_ts, init_conf[2:], grasp_states,
                             time_limit, cs_step_size, ts_step_size, goal_bias, goal_tol)
    if arm_plan is None:
        return None
    full_plan = np.hstack((np.tile(init_conf[:2], (arm_plan.shape[0], 1)), arm_plan))
    return full_plan


def _compute_n_reachable_grasps(base_pos, grasps, mobile_fmap: MobileFeasibilityMap):
    """
    Computes how many of the given grasp poses are reachable from the base position.
    """
    n_reachable = 0
    for grasp in grasps:
        if mobile_fmap.is_feasible(grasp, base_pos):
            n_reachable += 1
    return n_reachable


def _rank_base_candidates(base_candidates, grasps, mobile_fmap: MobileFeasibilityMap):
    """
    Ranks the base candidates based on the number of reachable grasps.
    """
    scores = np.zeros(len(base_candidates))
    for i, base_pos in enumerate(base_candidates):
        scores[i] = _compute_n_reachable_grasps(base_pos, grasps, mobile_fmap)
    return np.argsort(scores)[::-1]  # sort in descending order


def fm_rrt(cs: ConfigurationSpace, ts: TaskSpace, mobile_fmap: MobileFeasibilityMap, init_conf: np.ndarray,
           compact_arm_conf: np.ndarray, grasps: np.ndarray, time_limit=30, base_time_limit=1, n_base_candidates=5,
           base_epsilon=0.2, base_step_size=0.3, arm_cs_step_size=0.3, arm_jplus_step_size=0.3, goal_bias=0.1,
           goal_tolerance=0.05) -> Optional[np.ndarray]:
    """
    Planer for a mobile manipulator performing a grasping task.

    Works in 3 stages:
        1. Plans an initial path from `init_conf` to a predefined "low-volume" arm configuration `compact_arm_conf` using Greedy-RRT.
        2. Searches for a feasible base position from which the manipulator can reach the object (checked using the Feasibility Map).
        3. Plans an arm-only path (with fixed base) to the goal grasp using J+-RRT.

    Args:
        cs (ConfigurationSpace): Full configuration space of the mobile manipulator (base + arm). It is assumed that the
            first two coordinates correspond to the base position (x, y) in XY plane and the third controls the base joint of the arm.
        ts (TaskSpace): Task space of the mobile manipulator.
        mobile_fmap (MobileFeasibilityMap): Feasibility map for the manipulator.
        init_conf (np.ndarray): Full configuration of the robot's initial state.
        compact_arm_conf (np.ndarray): "Low-volume" compact arm configuration from which the second stage of the algorithm begins.
        grasps (np.ndarray): Array of grasp poses as homogeneous matrices (shape: (N, 4, 4)).
        time_limit (float): Total allowed time for planning (in seconds).
        base_time_limit (float): Time to continue planning (in the second stage) after the first base solution is found.
        n_base_candidates (int): Number of base candidates to generate. From these candidates, the best one is picked 
            (based on number of reachable grasps) for stage 3.
        base_epsilon (float): Minimum distance required between base positions.
        base_step_size (float): Step size for base-space RRT expansion.
        arm_cs_step_size (float): Step size for J+-RRT arm-space expansion.
        arm_jplus_step_size (float): Step size for J+-RRT task space expansion.
        goal_bias (float): Probability of sampling a goal state during arm planning (J+-RRT).
        goal_tolerance (float): Goal tolerance in task space.

    Returns:
        np.ndarray: Result path as an array of waypoints (shape: (number of waypoints, DoF)), or None if planning failed.
    """
    grasp_states = _precompute_grasps_states(grasps)
    start_time = time.time()

    plan_to_compact_conf = plan_arm_motion_to_compact_conf(
        cs, init_conf, compact_arm_conf, time_limit, arm_cs_step_size)
    if plan_to_compact_conf is None:
        return None

    remaining_time = time_limit - (time.time() - start_time)
    base_candidates, base_plans = find_base_positions(cs, mobile_fmap, np.hstack((init_conf[:2], compact_arm_conf)), grasps, n_base_candidates, remaining_time,
                                                      base_time_limit, base_epsilon, base_step_size)
    if len(base_candidates) == 0:
        return None

    rank_indices = _rank_base_candidates(base_candidates, grasps, mobile_fmap)
    idx = 0  # pick the best candidate
    arm_plan = plan_arm_motion(cs, ts, base_candidates[rank_indices[idx]], grasp_states,
                               time_limit - (time.time() - start_time), arm_cs_step_size, arm_jplus_step_size,
                               goal_bias, goal_tolerance)
    
    if arm_plan is not None:
        # found solution
        return np.vstack((plan_to_compact_conf[:-1, :], base_plans[rank_indices[idx]], arm_plan[1:, :]))

    return None
