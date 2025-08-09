# This is a modified version of the "rm4d/reachability_map.py" file from the RM4D library. See the README.md file for more details.
#
# RM4D is licensed under the MIT license.
# The original license text is retained in the LICENSE_RM4D file

import burg_toolkit as burg
import numpy as np
from abc import ABC
from typing import Callable

from .fmap_construction import FeasibilityMapConstructorBase
from .fmap_base import FeasibilityMapBase


class FeasibilityMap4DBase(ABC):
    """
    Holds common information about the FM4D map structure.

    This map structure is based on:
    Rudorfer, RM4D: A Combined Reachability and Inverse Reachability Map for Common 6-/7-axis Robot Arms by Dimensionality Reduction to 4D

    Args:
        xy_limits (list[float]): [min, max] bounds for the x and y position dimensions.
        z_limits (list[float]): [min, max] bounds for the z (height) dimension.
        voxel_res (float): Resolution of the voxel grid. Defaults to 0.05.
        n_bins_theta (int): Number of bins used to discretize the theta angle (the inclination from the world Z-axis).
            Defaults to 36.
    """

    def __init__(self, xy_limits=None, z_limits=None, voxel_res=0.05, n_bins_theta=36):
        if xy_limits is None:
            xy_limits = [-1.05, 1.05]
        if z_limits is None:
            z_limits = [0, 1.35]

        # [min, max]
        self.xy_limits = xy_limits
        self.z_limits = z_limits
        self.theta_limits = [0, np.pi]

        # get dimensions
        self.voxel_res = voxel_res
        self.n_bins_xy = int(np.ceil((self.xy_limits[1] - self.xy_limits[0])/voxel_res))
        self.n_bins_z = int(np.ceil((self.z_limits[1] - self.z_limits[0])/voxel_res))
        self.n_bins_theta = n_bins_theta

        # check achieved resolution
        assert np.isclose((self.xy_limits[1] - self.xy_limits[0])/self.n_bins_xy, self.voxel_res)
        assert np.isclose((self.z_limits[1] - self.z_limits[0])/self.n_bins_z, self.voxel_res)
        self.theta_res = (self.theta_limits[1] - self.theta_limits[0])/self.n_bins_theta

    def _get_p_z(self, tf_ee):
        """
        Gets the z-coordinate of the EE position.
        """
        return tf_ee[2, 3]

    def _get_rotation_2d(self, tf_ee):
        """
        Gives the rotation that aligns tf_ee such that its z-axis is in the x+z plane as 2d rotation matrix.
        """
        rz_x, rz_y = tf_ee[:2, 2]  # first two components of the z-axis
        # get angle of rotation
        psi = np.arctan2(rz_y, rz_x)
        # build inverse rotation matrix to rotate back by psi
        rot_mat_2d = np.array([
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)]
        ])

        return rot_mat_2d

    def _get_theta(self, tf_ee):
        """
        Gets the angle between EE's r_z and the world z axis in rad.
        """
        # dot product: [0, 0, 1] dot [rz_x, rz_y, rz_z] -- simplifies to rz_z
        rz_z = tf_ee[2, 2]
        theta = np.arccos(rz_z)
        return theta

    def _get_canonical_base_position(self, tf_ee):
        """
        Calculates (x*, y*) for a given EE pose.
        """
        p_x, p_y = tf_ee[:2, 3]
        rot2d = self._get_rotation_2d(tf_ee)
        x_star, y_star = rot2d @ np.array([-p_x, -p_y])
        return x_star, y_star

    def _get_z_index(self, p_z):
        """
        Given a p_z, gives the corresponding index in the map.
        """
        z_idx = int((p_z - self.z_limits[0]) / self.voxel_res)
        if z_idx < 0:
            raise IndexError(f'z idx < 0 -- {p_z}')
        if z_idx >= self.n_bins_z:
            raise IndexError(f'z idx too large -- {p_z}')
        return z_idx

    def _get_theta_index(self, theta):
        """
        Given the value of theta, gives the corresponding index.
        """
        # if theta is pi, we want it to be included in the last bin
        if np.isclose(theta, np.pi):
            return self.n_bins_theta - 1

        theta_idx = int((theta - self.theta_limits[0]) / self.theta_res)
        if theta_idx < 0:
            raise IndexError(f'theta idx < 0 -- {theta}')
        if theta_idx >= self.n_bins_theta:
            raise IndexError(f'theta idx too large -- {theta}')
        return theta_idx

    def _get_xy_index(self, xy):
        """
        Given x* or y* from the canonical base position, gives the corresponding index.
        """
        x_idx = int((xy - self.xy_limits[0]) / self.voxel_res)
        if x_idx < 0:
            raise IndexError(f'xy_idx < 0 -- {xy}')
        if x_idx >= self.n_bins_xy:
            raise IndexError(f'xy idx too large -- {xy}')
        return x_idx

    def get_indices_for_ee_pose(self, tf_ee: np.ndarray) -> tuple[int, int, int, int]:
        """
        Computes map indices corresponding to a given end-effector pose.

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.

        Returns:
            tuple[int, ...]: Indices in the feasibility map corresponding to the pose.

        Raises:
            IndexError: If tf_ee is outside the bounds of the map.
        """
        # perform the dimensionality reduction
        p_z = self._get_p_z(tf_ee)
        theta = self._get_theta(tf_ee)
        x_star, y_star = self._get_canonical_base_position(tf_ee)

        # determine indices
        z_idx = self._get_z_index(p_z)
        theta_idx = self._get_theta_index(theta)
        x_idx = self._get_xy_index(x_star)
        y_idx = self._get_xy_index(y_star)

        return z_idx, theta_idx, x_idx, y_idx


class FeasibilityMap4DConstructor(FeasibilityMap4DBase, FeasibilityMapConstructorBase):
    """
    Constructor for the 4D feasibility map. Uses a flexible but memory-inefficient data structure to support iterative map
    construction. Once construction is finished, the map should be converted into a `FeasibilityMap4D` instance for more
    efficient memory usage.

    FM4D assumes that each configuration vector begins with the base joint and ends with the wrist joint. These two
    joints are excluded from storage during construction. It is also assumed that joint values 
    lie within the range [-π, π].

    Args:
        xy_limits (list[float]): [min, max] bounds for the x and y position dimensions.
        z_limits (list[float]): [min, max] bounds for the z (height) dimension.
        voxel_res (float): Resolution of the voxel grid. Defaults to 0.05.
        n_bins_theta (int): Number of bins used to discretize the theta angle (the inclination from the world Z-axis).
            Defaults to 36.
        configuration_len (int): Length of the full configuration vector (i.e., number of DoF). 
    """

    def __init__(self, xy_limits=None, z_limits=None, voxel_res=0.05, n_bins_theta=36, configuration_len=7):
        super().__init__(xy_limits, z_limits, voxel_res, n_bins_theta)
        self.map = {}
        self.n_configurations = 0
        self.configuration_len = configuration_len - 2  # FM4D does not store wrist and base joint

    def _is_similar(self, query_conf, confs, tol=0.5):
        """
        Checks if the given configuration is similar to any of the existing ones.
        """
        dists = np.linalg.norm(confs - query_conf, axis=1)
        if np.any(dists < tol):
            return True

        return False

    def add_conf(self, map_indices: tuple[int, ...], configuration: np.ndarray):
        """
        Adds a new configuration to the feasibility map at the specified indices (if a similar configuration has not already
        been stored). In FM4D, the base and wrist joints are excluded before storage.

        Args:
            map_indices (tuple[int, ...]): Indices into the feasibility map.
            configuration (np.ndarray): Full configuration vector (shape: (DoF,)).
        """
        configuration = configuration[1:-1]  # excluding base and wrist joints
        if map_indices not in self.map:
            self.map[map_indices] = np.array(configuration, dtype=np.float32).reshape((1, -1))
            self.n_configurations += 1
        else:
            old_confs = self.map[map_indices]
            if not self._is_similar(configuration, old_confs):
                # add to the map
                new_confs = np.vstack((old_confs, configuration))
                self.map[map_indices] = new_confs
                self.n_configurations += 1

    def finalize_map(self) -> "FeasibilityMap4D":
        """
        Finalizes the feasibility map construction and returns a more memory-efficient `FeasibilityMap4D`.

        Returns:
            FeasibilityMap4D: The finalized feasibility map instance.
        """
        # map stores slice indices for the configurations array
        map = np.full(
            shape=(self.n_bins_z, self.n_bins_theta, self.n_bins_xy, self.n_bins_xy, 2),  # 2 for low and high index
            fill_value=-1,
            dtype=np.int32
        )
        # all of the configurations stored in one array
        configurations = np.empty(
            shape=(self.n_configurations, self.configuration_len),
            dtype=np.float32
        )

        curr_idx = 0
        for indices, confs in self.map.items():
            start = curr_idx
            end = curr_idx + confs.shape[0]

            z, t, x, y = indices
            map[z, t, x, y, 0] = start
            map[z, t, x, y, 1] = end
            configurations[start:end, :] = confs

            curr_idx += confs.shape[0]

        return FeasibilityMap4D(map, configurations, self.xy_limits, self.z_limits, self.voxel_res, self.n_bins_theta)

    def to_file(self, filename: str):
        """
        Saves the current state of the map constructor to a ".npy" file.

        Args:
            filename (str): Path to the file. Should end with ".npy".
        """
        save_dict = {
            'map': self.map,
            'xy_limits': self.xy_limits,
            'z_limits': self.z_limits,
            'voxel_res': self.voxel_res,
            'n_bins_theta': self.n_bins_theta,
            'n_configurations': self.n_configurations,
            'configuration_len': self.configuration_len
        }
        np.save(filename, save_dict)

    @classmethod
    def from_file(cls, filename: str) -> "FeasibilityMap4DConstructor":
        """
        Loads a `FeasibilityMap4DConstructor` instance from a ".npy" file.

        Args:
            filename (str): Path to the saved `.npy` file.

        Returns:
            FeasibilityMap4DConstructor: Saved `FeasibilityMap4DConstructor` instance.
        """
        d = np.load(filename, allow_pickle=True).item()
        fm = cls(xy_limits=d['xy_limits'],
                 z_limits=d['z_limits'],
                 voxel_res=d['voxel_res'],
                 n_bins_theta=d['n_bins_theta'],
                 configuration_len=d['configuration_len'])
        fm.map = d['map']
        fm.n_configurations = d['n_configurations']

        print(f'{cls.__name__} loaded from {filename}')
        return fm


class FeasibilityMap4D(FeasibilityMap4DBase, FeasibilityMapBase):
    """
    Memory-efficient feasibility map based on RM4D by Martin Rudorfer.

    Unlike RM4D, our map stores the robot's joint configurations (found during construction) that lead to the corresponding cell in the map.

    In this feasibility map, the stored configurations exclude the robot's base and wrist joints. These joints
    are reconstructed on-the-fly based on the desired end-effector pose.

    This map should be constructed using a `FeasibilityMap4DConstructor` followed by its `finalize_map()` method.

    Args:
        map (np.ndarray): 5D array of shape (n_bins_z, n_bins_theta, n_bins_x, n_bins_y, 2),
            storing index slices into `configurations`.
        configurations (np.ndarray): Configuration stored in the map (shape: (n_configurations, DoF - 2)) excluding the
            base and wrist joints.
        xy_limits (list[float]): [min, max] bounds for the x and y position dimensions.
        z_limits (list[float]): [min, max] bounds for the z (height) dimension.
        voxel_res (float): Resolution of the voxel grid. Defaults to 0.05.
    """

    def __init__(self, map: np.ndarray, configurations: np.ndarray, xy_limits=None, z_limits=None, voxel_res=0.05, n_bins_theta=36):
        super().__init__(xy_limits, z_limits, voxel_res, n_bins_theta)
        self.map = map
        self.configurations = configurations

    def to_file(self, filename):
        """
        Saves the feasibility map to a ".npy" file.

        Args:
            filename (str): Path to the file. Should end with ".npy".
        """
        save_dict = {
            'map': self.map,
            'configurations': self.configurations,
            'xy_limits': self.xy_limits,
            'z_limits': self.z_limits,
            'voxel_res': self.voxel_res,
            'n_bins_theta': self.n_bins_theta,
        }
        np.save(filename, save_dict)

    @classmethod
    def from_file(cls, filename) -> "FeasibilityMap4D":
        """
        Loads a `FeasibilityMap4D` instance from a ".npy" file.

        Args:
            filename (str): Path to the saved ".npy" file.

        Returns:
            FeasibilityMap4D: The loaded feasibility map instance.
        """
        d = np.load(filename, allow_pickle=True).item()
        fm = cls(d['map'],
                 d['configurations'],
                 d['xy_limits'],
                 d['z_limits'],
                 d['voxel_res'],
                 d['n_bins_theta'])

        fm.map = d['map']
        print(f'{cls.__name__} loaded from {filename}')
        return fm

    def get_configurations(self, map_indices: tuple[int, ...]) -> np.ndarray:
        """
        Retrieves the configurations stored in the map at the specified indices.
        These configurations are missing base and wrist joints.

        Args:
            map_indices (tuple[int, ...]): Indices into the feasibility map.

        Returns:
            np.ndarray: Array of stored configurations (shape: (n, DoF - 2)),
            or None if no configurations are stored at the given indices.
        """
        low, high = self.map[map_indices]
        if low == -1:
            return None
        return self.configurations[low:high, :]

    def _get_wrist_joint(self, tf_ee, joints, forward_kinematics):
        """
        Estimates the wrist joint angle needed to align the end-effector's x-axis.
        """
        x_axis = tf_ee[:3, 0]
        pos_quat = forward_kinematics(np.hstack((joints, [0])))
        ee_zero = burg.util.tf_from_pos_quat(pos_quat[:3], pos_quat[3:], convention='pybullet')
        rot = ee_zero[:3, :3]
        x_proj = np.linalg.inv(rot) @ x_axis
        # compute the signed angle
        return np.arctan2(x_proj[1], x_proj[0])  # implicit projection onto XY plane

    def _get_base_joint(self, tf_ee, joints, forward_kinematics):
        """
        Estimates the base joint value required to match the desired EE position.
        """
        # ee_pose with zero base joint (and zero wrist joint but that does not matter)
        xy_now = forward_kinematics(np.hstack((0, joints, 0)))[:2]  # implicit projection
        xy_target = tf_ee[:2, 3]  # desired EE position
        return np.arctan2(xy_now[0] * xy_target[1] - xy_now[1] * xy_target[0],
                          xy_now[0] * xy_target[0] + xy_now[1] * xy_target[1])

    def get_full_configurations(self, tf_ee: np.ndarray, forward_kinematics: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Retrieves full robot configurations corresponding to the given end-effector pose.

        Reconstructs the complete configuration vectors, including joints omitted during storage (base and wrist joints).

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.
            forward_kinematics (Callable): Function that maps a configuration to a task-space pose (a NumPy array 
                [x, y, z, h_x, h_y, h_z, h_w], with the last four elements forming a unit quaternion).

        Returns:
            np.ndarray: Array of stored robot configurations (shape: (n, DoF)).
        """
        try:
            query_indices = self.get_indices_for_ee_pose(tf_ee)
            confs = self.get_configurations(query_indices)
            if confs is None:
                return np.array([])

            full_confs = []
            for conf in confs:
                base_joint = self._get_base_joint(tf_ee, conf, forward_kinematics)
                joints_with_base = np.hstack((base_joint, conf))
                wrist_joint = self._get_wrist_joint(tf_ee, joints_with_base, forward_kinematics)
                full_conf = np.hstack((joints_with_base, wrist_joint))
                full_confs.append(full_conf)

            return np.array(full_confs)

        except IndexError:
            return np.array([])
