from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import burg_toolkit as burg
from time import time

from .jogramop.scenario import Scenario
from ..spaces import ConfigurationSpace, TaskSpace
from ..planners import j_plus_rrt, fm_rrt
from ..latent import latent_j_plus_rrt, Autoencoder
from ..fmap import FeasibilityMap4D, MobileFeasibilityMap
from .visualization import JogramopVisualizer

COLLISION_STEP_SIZE = 0.1
CS_STEP_SIZE = 0.4
J_PLUS_STEP_SIZE = 0.3
LATENT_STEP_SIZE = 0.03
GOAL_TOLERANCE = 0.05
GOAL_BIAS = 0.1


class JogramopBenchmark:
    """
    Holds benchmark data.
    """

    def __init__(self, algorithm, scenario_id, runtimes, time_limit, plans):
        self.algorithm = algorithm
        self.scenario_id = scenario_id
        self.runtimes = np.array(runtimes)
        self.time_limit = time_limit
        self.plans = plans  # python array of numpy arrays (plans)

    def save(self, filename: str):
        """
        Saves the benchmark data into a file.

        Args:
            filename: path to file (without .pkl extension)
        """
        f = open(f"{filename}.pkl", "wb")
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(filename: str) -> 'JogramopBenchmark':
        """
        Loads a benchmark from a file.

        Args:
            filename: path to file
        """
        f = open(filename if filename.endswith(".pkl") else f"{filename}.pkl", "rb")
        benchmark = pickle.load(f)
        f.close()
        return benchmark


def plot_success_curves(filename: str, benchmarks: List[JogramopBenchmark], labels: Optional[List[str]] = None, n_bins: int = 60):
    """
    Plots success curves for a list of benchmark results and saves the plot as a PDF file.

    Args:
        filename (str): Name of the output file (without ".pdf" extension).
        benchmarks (List[JogramopBenchmark]): List of benchmark results to include in the plot.
        labels (Optional[List[str]]): Custom labels for the legend. If None, algorithm names from the benchmarks are used.
        n_bins (int): Number of time intervals over which the success rate is computed.
    """

    plt.figure(figsize=(8, 5))
    max_time = np.array([b.time_limit for b in benchmarks]).max()
    time_bins = np.linspace(0, max_time, n_bins)

    # line_styles = ['-', '--', ':', '-.']

    for i, b in enumerate(benchmarks):
        probabilities = [(b.runtimes <= t).sum() / b.runtimes.shape[0] for t in time_bins]
        # style = line_styles[i % len(line_styles)]
        label = b.algorithm if labels is None else labels[i]
        plt.plot(time_bins, probabilities, label=label, linewidth=3)

    font_size = 20
    plt.xlabel("Runtime [s]", fontsize=font_size)
    plt.ylabel("Probability of Finding a Solution", fontsize=font_size)
    plt.xlim(0, max_time)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(fontsize=font_size * 0.8)
    plt.tick_params(axis='both', labelsize=font_size * 0.8)
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')


def visualize_plans(benchmark: JogramopBenchmark):
    """
    Visualizes plans from the benchmark.

    Args:
        benchmark (JogramopBenchmark): Benchmark to visualize.
    """
    v = JogramopVisualizer(Scenario(benchmark.scenario_id), draw_debug_lines=True)
    for plan in benchmark.plans:
        v.reset()
        if plan is None:
            print("Encountered None plan (this benchmark run did not found a feasible plan).")
        else:
            v.draw_plan(plan, visualize_nodes=True)
        input("Press enter to show next plan.")
    print("All plans were visualized. Bye.")
    v.dismiss()


def _precompute_grasps_states(grasps_tf):
    grasp_states = np.empty((grasps_tf.shape[0], 7))
    for i, tf in enumerate(grasps_tf):
        pos, quat = burg.util.position_and_quaternion_from_tf(tf, convention='pybullet')
        grasp_states[i, :] = np.hstack((pos, quat))
    return grasp_states


def _initialize_benchmark(scenario_id: int, fmap_filename: str, model_filename: str):
    s = Scenario(scenario_id)
    robot, sim = s.get_robot_and_sim(with_gui=False)
    start_conf = np.array(robot.home_conf)
    grasps_tf = s.grasp_poses

    grasp_states = _precompute_grasps_states(grasps_tf)

    def collision_checker(c):
        robot.reset_arm_joints(c)
        return robot.in_collision() or robot.in_self_collision()

    def self_collision_checker(c):
        robot.reset_arm_joints(c)
        return robot.in_self_collision()

    cs = ConfigurationSpace(robot.arm_joint_limits(), collision_checker, collision_step_size=COLLISION_STEP_SIZE)

    def fk(conf):
        pos, orn = robot.forward_kinematics(conf)
        return np.hstack((pos, orn))

    def ik(state):
        return robot.inverse_kinematics(state[:3], state[3:], null_space_control=True)

    def jacobian(conf):
        robot.reset_arm_joints(conf)
        q = list(robot.joint_pos())  # get movable joints (includes gripper joints)
        num_joints = len(q)
        J_linear, J_angular = robot.bullet_client.calculateJacobian(robot.body_id, robot.end_effector_id,
                                                                    [0, 0, 0], q, [0]*num_joints, [0]*num_joints)
        J = np.vstack((J_linear, J_angular))[:, :-2]  # get rid of gripper joints
        return J

    ts = TaskSpace(limits=np.array([[-0.6, 2.6], [-1.6, 1.6], [0.0, 1.0]]),
                   forward_kinematics=fk, inverse_kinematics=ik, jacobian=jacobian,
                   init_base_to_world_tf=s.default_robot_pose(), rot_scale=0.01)

    fm = FeasibilityMap4D.from_file(fmap_filename)

    def get_robot_base_pose(conf):
        robot.reset_arm_joints(conf)
        pos, quat, *_ = robot.bullet_client.getLinkState(robot.body_id, 1)
        base_pos = burg.util.tf_from_pos_quat(pos, quat, convention='pybullet')
        return base_pos

    construction_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.05],
        [0.0, 0.0, 0.0, 1.0],
    ])

    mobile_fmap = MobileFeasibilityMap(fm, collision_checker, fk, get_robot_base_pose, construction_pose)

    low_volume_conf = np.array([0, -1.7, -2.4, -3.11, -0.3, 1.54, 0.7534486589746342])

    model = Autoencoder.from_file(model_filename)

    return cs, ts, start_conf, grasps_tf, grasp_states, mobile_fmap, low_volume_conf, model, self_collision_checker


def benchmark(scenario_id: int, algorithm: str, n_runs: int, time_limit: float, fmap_filename: str, model_filename: str,
              out_filename: str = None):
    """
    Runs benchmark of a path planning algorithm in a scenario from the Jogramop framework. Runs the selected algorithm multiple times
    and records the resulting plans and runtimes.

    Args:
        scenario_id (int): ID of the Jogramop scenario.
        algorithm (str): Planning algorithm to run. Currently supported options: 'J+-RRT', 'FM-RRT', 'Latent-RRT'.
        n_runs (int): Number of times the planner will be executed on the scenario.
        time_limit (float): Maximum time allowed per planning run (in seconds).
        fmap_filename (str): Path to the feasibility map file used in FM-based planning algorithms.
        model_filename (str): Path to the `Autoencoder` model used by the Latent-RRT algorithm.
        out_filename (str): If provided, saves the benchmark results to this file.

    Returns:
        JogramopBenchmark: Benchmark results.
    """

    cs, ts, start_conf, grasps_tf, grasp_states, mobile_fmap, low_volume_conf, model, self_collision_checker = _initialize_benchmark(
        scenario_id, fmap_filename, model_filename)

    match algorithm:
        case 'J+-RRT':
            def planner(): return j_plus_rrt(cs, ts, start_conf, grasp_states, time_limit, CS_STEP_SIZE, J_PLUS_STEP_SIZE,
                                             GOAL_BIAS, GOAL_TOLERANCE)
        case 'FM-RRT':
            def planner(): return fm_rrt(cs, ts, mobile_fmap, start_conf, low_volume_conf, grasps_tf, time_limit,
                                         time_limit/10, 5, 0.3, 0.4, CS_STEP_SIZE, J_PLUS_STEP_SIZE, GOAL_BIAS,
                                         GOAL_TOLERANCE)
        case 'Latent-J+-RRT':
            def planner(): return latent_j_plus_rrt(cs, ts, model, self_collision_checker, start_conf, grasp_states, time_limit,
                                                    LATENT_STEP_SIZE, J_PLUS_STEP_SIZE, GOAL_BIAS, GOAL_TOLERANCE)

        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    plans = []
    runtimes = []

    for i in tqdm(range(n_runs), desc=f"{algorithm} benchmark"):
        start_time = time()
        result = planner()
        plan = result[0] if isinstance(result, tuple) else result
        runtime = (time() - start_time) if plan is not None else float('inf')
        runtimes.append(runtime)
        plans.append(plan)

    benchmark = JogramopBenchmark(algorithm, scenario_id, runtimes, time_limit, plans)
    if out_filename is not None:
        benchmark.save(out_filename)

    return benchmark
