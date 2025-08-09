"""
Adds several visualizations tools to the jogramop framework.
"""
import PIL
import PIL.Image
import numpy as np

from .jogramop.scenario import Scenario
from .jogramop.visualization import indicate_workspace
from ..tree import Tree

GRAPH_COLOR = [87 / 255, 126 / 255, 137 / 255]
PLAN_COLOR = [225 / 255, 163 / 255, 111 / 255]  # [0.53, 0.905, 0.53]

PLAN_WIDTH = 0.003
GRAPH_WIDTH = 0.002

DEBUG_LINE_WIDTH = 7


class JogramopVisualizer:
    """
    Visualizes scenario from the Jogramop framework in pybullet simulation.

    Args:
        scenario (Scenario): jogramop.Scenario instance.
        draw_debug_lines (bool): Draws 2D debug lines instead of thin boxes - takes less time to visualize.
    """

    def __init__(self, scenario: Scenario, draw_debug_lines: bool = False):
        self.active = True
        self.scenario = scenario
        self.draw_debug_lines = draw_debug_lines

        self.robot, self.sim = scenario.get_robot_and_sim(with_gui=True)
        indicate_workspace(self.sim)
        self.video_id = None
        self.gripper_closed = False
        self.added_bodies = []

    def __del__(self):
        self.dismiss()

    def _draw_sphere(self, center, radius, color):
        visual_shape = self.sim.bullet_client.createVisualShape(
            self.sim.bullet_client.GEOM_SPHERE, radius=radius, rgbaColor=color + [1])
        id = self.sim.bullet_client.createMultiBody(0, -1, visual_shape, basePosition=center)
        self.added_bodies.append(id)
        return id

    def _draw_line(self, x, y, width, color):
        if self.draw_debug_lines:
            return self.sim.bullet_client.addUserDebugLine(x, y, color, DEBUG_LINE_WIDTH, 0)

        mid = (x + y) / 2
        length = np.linalg.norm(y - x)
        direction = (y - x) / length
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction))
        quat = self.sim.bullet_client.getQuaternionFromAxisAngle(axis, angle)
        visual_shape = self.sim.bullet_client.createVisualShape(
            self.sim.bullet_client.GEOM_BOX, halfExtents=[width, width, length / 2], rgbaColor=color + [1])
        body_id = self.sim.bullet_client.createMultiBody(
            0, -1, visual_shape, basePosition=mid.tolist(), baseOrientation=quat)
        self.added_bodies.append(body_id)
        return body_id

    def reload(self):
        """
        Reloads the simulation.
        """
        if self.active:
            self.sim.dismiss()

        self.robot, self.sim = self.scenario.get_robot_and_sim(with_gui=True)
        indicate_workspace(self.sim)
        self.active = True

    def dismiss(self):
        """
        Stops the simulation.
        """
        if self.active:
            self.sim.dismiss()
            self.active = False

    def reset(self):
        """
        Resets the simulation to its initial state (without reloading the window if not necessary).
        """
        if not self.active:
            self.reload()
        self.sim.bullet_client.removeAllUserDebugItems()  # remove all lines (tree or plans)
        while len(self.added_bodies) > 0:
            self.sim.bullet_client.removeBody(self.added_bodies.pop())  # removing spheres - takes eternity :(
        self.robot.reset_arm_joints(self.robot.home_conf)
        if self.gripper_closed:
            self.robot.open()
            self.gripper_closed = False

    def draw_plan(self, plan: np.ndarray, visualize_nodes: bool = False, color=PLAN_COLOR):
        """
        Visualizes robot's plan in the simulation.

        Args:
            plan (np.ndarray): Path planning plan (shape: (number of waypoints, DoF)).
            visualize_nodes (bool): Set to True if you want to also visualize Nodes.
        """
        if not self.active:
            self.reload()

        tmp_robot, tmp_sim = self.scenario.get_robot_and_sim(with_gui=False)
        coords = []
        for conf in plan:
            pos, _ = tmp_robot.forward_kinematics(conf)
            coords.append(pos)
        tmp_sim.dismiss()

        if visualize_nodes:
            self._draw_sphere(coords[0], 4 * PLAN_WIDTH, color)
        for i in range(1, len(coords)):
            self._draw_line(coords[i - 1], coords[i], PLAN_WIDTH, color)
            if visualize_nodes:
                self._draw_sphere(coords[i], 4 * PLAN_WIDTH, color)

    def draw_tree(self, tree: Tree, tree_type: str = 'cs', visualize_nodes: bool = False):
        """
        Visualized an RRT tree in the simulation.

        Args:
            tree (Tree): pp4rg.Tree instance.
            tree_type (str): Either 'cs' (tree grown in configuration space), 'ts' (tree grown in task space - e.g. 
                ts_rrt algorithm) or 'ls' (tree growth in latent space - e.g. latent_rrt algorithm) depending on
                the algorithm which generated the Tree. 
            visualize_nodes (bool): Set to True if you want to also visualize Nodes.
        """
        def _get_edges_rrt(tree):
            tmp_robot, tmp_sim = self.scenario.get_robot_and_sim(with_gui=False)
            edges = []
            for node in tree.nodes[1:]:  # skipping root
                parent_pos, _ = tmp_robot.forward_kinematics(node.parent.coords)
                node_pos, _ = tmp_robot.forward_kinematics(node.coords)
                edges.append((parent_pos, node_pos))
            tmp_sim.dismiss()
            return edges

        def _get_edges_ts_rrt(tree):
            edges = []
            for node in tree.nodes[1:]:  # skipping root
                edges.append((node.parent.coords[:3], node.coords[:3]))
            return edges

        def _get_edges_ls_rrt(tree):
            tmp_robot, tmp_sim = self.scenario.get_robot_and_sim(with_gui=False)
            edges = []
            for node in tree.nodes[1:]:  # skipping root
                parent_pos, _ = tmp_robot.forward_kinematics(node.parent.conf)
                node_pos, _ = tmp_robot.forward_kinematics(node.conf)
                edges.append((parent_pos, node_pos))
            tmp_sim.dismiss()
            return edges

        
        if not self.active:
            self.reload()
        
        # collect edges
        match tree_type:
            case 'cs':
                edges = _get_edges_rrt(tree)
            case 'ts':
                edges = _get_edges_ts_rrt(tree)
            case 'ls':
                edges = _get_edges_ls_rrt(tree)
            case _:
                raise ValueError("Only 'cs' and 'ts' tree types are supported in draw_tree.")

        # draw edges
        if visualize_nodes:
            self._draw_sphere(edges[0][0], 4 * GRAPH_WIDTH, GRAPH_COLOR)  # root node

        for edge in edges:
            x, y = edge
            if visualize_nodes:
                self._draw_sphere(y, 4 * GRAPH_WIDTH, GRAPH_COLOR)
            self._draw_line(x, y, GRAPH_WIDTH, GRAPH_COLOR)

    def animate_plan(self, plan: np.ndarray, grasp: bool = False, n_backward: int = 0):
        """
        Shows an animation of a robot's plan.

        Args:
            plan (np.ndarray): Path planning plan (shape: (number of waypoints, DoF)).
            grasp (bool): If set to True, tries to grasp the object at the end of the plan. Defaults to False.
            n_backward (int): Does n backward motions (from the end of the plan) after grasping (grasp must be set to true).
        """

        if not self.active:
            self.reload()

        self.robot.reset_arm_joints(plan[0, :])
        for i in range(plan.shape[0]):
            self.robot.move_to(plan[i, :])

        if grasp:
            self.robot.close()
            self.gripper_closed = True
            for i in range(1, n_backward + 1):
                self.robot.move_to(plan[-i, :])

    def start_recording(self, filename: str):
        """
        Starts recording an MP4 video of the simulation.

        Args:
            filename (str): Output filename.
        """
        if not self.active:
            raise RuntimeError("Cannot start recording: Simulation is not running.")
        if self.video_id is not None:
            print("WARNING: Cannot start recording. Already recording the simulation.")
            return
        self.video_id = self.sim.bullet_client.startStateLogging(
            self.sim.bullet_client.STATE_LOGGING_VIDEO_MP4, f"{filename}.mp4")

    def stop_recording(self):
        """
        Stops the recording of the simulation and saves the result MP4 file.
        """
        if not self.active:
            raise RuntimeError("Cannot stop recording: Simulation is not running.")
        if self.video_id is None:
            print("WARNING: Cannot stop recording. There is no active recording.")
            return
        self.sim.bullet_client.stopStateLogging(self.video_id)
        self.video_id = None

    def take_picture(self, filename: str, shadows: bool = True):
        """
        Saves a screenshot of the simulator.

        Args:
            filename (str): Output filename.
            shadows (bool): Won't render shadows if set to False.
        """
        if not self.active:
            raise RuntimeError("Cannot take a picture: Simulation is not running.")

        width, height, view_matrix, proj_matrix, _, _, _, _, _, _, _, _ = self.sim.bullet_client.getDebugVisualizerCamera()
        _, _, img_array, _, _ = self.sim.bullet_client.getCameraImage(width=width,
                                                                      height=height,
                                                                      viewMatrix=view_matrix,
                                                                      projectionMatrix=proj_matrix,
                                                                      shadow=shadows)
        img = PIL.Image.fromarray(img_array)
        img.save(f"{filename}.png")
