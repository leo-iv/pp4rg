# This is a modified version of the "rm4d/zacharias_map.py" file from the RM4D library. See the README.md file for more details.
#
# RM4D is licensed under the MIT license.
# The original license text is retained in the LICENSE_RM4D file.

import numpy as np
from abc import ABC
import burg_toolkit as burg
from typing import Callable

from .fmap_base import FeasibilityMapBase
from .fmap_construction import FeasibilityMapConstructorBase


class FeasibilityMap6DBase(ABC):
    """
    Holds common information about the FM6D map structure.

    This map structure is based on:
    Zacharias et al., The Capability Map: A Tool to Analyze Robot Arm Workspaces, International Journal of Humanoid Robotics, 2013.

    Args:
        xy_limits (list[float]): [min, max] bounds for the x and y position dimensions.
        z_limits (list[float]): [min, max] bounds for the z (height) dimension.
        voxel_res (float): Resolution of the voxel grid. Defaults to 0.05.
        n_sphere_points (int): Number of discrete sphere points (used to discretize the approach direction).
        n_bins_in_plane (int): Number of bins used for the discretization of the EE in-plane rotation.
        sphere_point_tfs (np.ndarray): Optional array of shape (n_sphere_points, 4, 4) containing homogeneous 
            transformation matrices that define the approach directions. If None, they will be automatically generated 
            according to the method described by Zacharias et al.
    """

    def __init__(self, xy_limits=None, z_limits=None, voxel_res=0.05, n_sphere_points=200, n_bins_in_plane=12,
                 sphere_point_tfs=None):
        if xy_limits is None:
            xy_limits = [-1.05, 1.05]
        if z_limits is None:
            z_limits = [0, 1.35]

        # [min, max]
        self.xy_limits = xy_limits
        self.z_limits = z_limits
        self.in_plane_limits = [0, 2*np.pi]

        # get dimensions
        self.voxel_res = voxel_res
        self.n_bins_xy = int(np.ceil((self.xy_limits[1] - self.xy_limits[0]) / voxel_res))
        self.n_bins_z = int(np.ceil((self.z_limits[1] - self.z_limits[0]) / voxel_res))
        self.n_sphere_points = n_sphere_points
        self.n_bins_in_plane = n_bins_in_plane

        # check achieved resolution
        assert np.isclose((self.xy_limits[1] - self.xy_limits[0]) / self.n_bins_xy, self.voxel_res)
        assert np.isclose((self.z_limits[1] - self.z_limits[0]) / self.n_bins_z, self.voxel_res)
        self.in_plane_res = (self.in_plane_limits[1] - self.in_plane_limits[0]) / self.n_bins_in_plane

        # generate sphere points for cache
        if sphere_point_tfs is None:
            self.sphere_point_tfs = self._get_sphere_point_tfs()
        else:
            # using provided points
            assert len(sphere_point_tfs) == self.n_sphere_points
            self.sphere_point_tfs = sphere_point_tfs

    def _get_sphere_point_tfs(self, n_points=None, radius=None):
        """
        We uniformly sample points on a sphere (more or less), and then build a tf for each of them. The z-axis
        looks towards the sphere centre, x and y are chosen arbitrarily.
        Spiral point algorithm as described in Saff, Kuijlaars, "Distributing many points on a sphere", 1997.
        https://perswww.kuleuven.be/~u0017946/publications/Papers97/art97a-Saff-Kuijlaars-MI/Saff-Kuijlaars-MathIntel97.pdf
        """
        if n_points is None:
            n_points = self.n_sphere_points
        if radius is None:
            radius = self.voxel_res / 2

        theta = np.empty(n_points, dtype=float)  # theta: [0, pi]
        phi = np.empty(n_points, dtype=float)  # phi:  [0, 2pi]
        for i in range(n_points):
            h = -1.0 + 2.0 * i / (n_points - 1.0)
            theta[i] = np.arccos(h)
            if i in [0, n_points - 1]:
                phi[i] = 0
            else:
                phi[i] = (phi[i - 1] + 3.6 / np.sqrt(n_points) * 1.0 / np.sqrt(1 - h ** 2)) % (2 * np.pi)

        # position on surface of sphere
        tfs = np.full((n_points, 4, 4), np.eye(4))
        tfs[:, 0, 3] = radius * np.sin(theta) * np.cos(phi)
        tfs[:, 1, 3] = radius * np.sin(theta) * np.sin(phi)
        tfs[:, 2, 3] = radius * np.cos(theta)

        # z-axis orientated towards origin
        z_axes = -tfs[:, :3, 3]  # negative point coordinates
        z_axes = z_axes / np.linalg.norm(z_axes, axis=-1)[:, np.newaxis]

        # arbitrarily choose x and y axes to form coordinate system
        def get_random_unit_vec():
            vec = np.random.normal(size=3)
            mag = np.linalg.norm(vec)
            if mag <= 1e-5:
                return get_random_unit_vec()
            else:
                return vec / mag

        # find some y orthogonal to z by using cross product with a random unit vector
        tmp_vec = get_random_unit_vec()
        y_axes = np.cross(z_axes, tmp_vec)
        while (np.linalg.norm(y_axes, axis=-1) == 0).any():
            tmp_vec = get_random_unit_vec()
            y_axes = np.cross(z_axes, tmp_vec)
            print('note: repeatedly attempting cross product to get sphere points.')

        # ensure normalized
        y_axes = y_axes / np.linalg.norm(y_axes, axis=-1)[:, np.newaxis]

        # now get x-axis
        x_axes = np.cross(y_axes, z_axes)
        x_axes = x_axes / np.linalg.norm(x_axes, axis=-1)[:, np.newaxis]

        tfs[:, :3, 0] = x_axes
        tfs[:, :3, 1] = y_axes
        tfs[:, :3, 2] = z_axes

        return tfs

    def _get_xy_index(self, xy):
        """
        Given EE x or y coordinate, gives the corresponding index in map.
        """
        x_idx = int((xy - self.xy_limits[0]) / self.voxel_res)
        if x_idx < 0:
            raise IndexError(f'xy_idx < 0 -- {xy}')
        if x_idx >= self.n_bins_xy:
            raise IndexError(f'xy idx too large -- {xy}')
        return x_idx

    def _get_z_index(self, z):
        """
        Given EE z coordinate, gives the corresponding index in the map.
        """
        z_idx = int((z - self.z_limits[0]) / self.voxel_res)
        if z_idx < 0:
            raise IndexError(f'z idx < 0 -- {z}')
        if z_idx >= self.n_bins_z:
            raise IndexError(f'z idx too large -- {z}')
        return z_idx

    def _get_sphere_point_index(self, tf_ee):
        """
        Finds the index of the sphere point with the most similar approach vector.
        I.e., the smallest angle between z-axis of the EE pose and z-axis of the sphere point frame.
        According to Eq. (34) in Zacharias et al.
        """
        query_r_z = tf_ee[:3, 2]
        spheres_r_z = self.sphere_point_tfs[:, :3, 2]

        idx = np.argmin(np.arccos(np.clip(np.dot(spheres_r_z, query_r_z), -1, 1)))
        return idx

    def _get_in_plane_rotation_index(self, tf_ee, sphere_point_idx):
        """
        Finds the index for the in-plane rotation according to equations (35)-(38) in Zacharias et al., 2013
        """
        x_axis = tf_ee[:3, 0]
        rot = self.sphere_point_tfs[sphere_point_idx][:3, :3]
        p_xy = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        # project x-axis onto xy plane
        x_hat = rot @ p_xy @ np.linalg.inv(rot) @ x_axis
        beta = np.arccos(np.clip(np.dot(x_hat / np.linalg.norm(x_hat), rot[:3, 0]), -1, 1))
        idx = int(np.floor(beta / self.in_plane_res + 0.5))
        if idx < 0:
            raise IndexError(f'in-plane rot index < 0 -- {np.rad2deg(beta)} deg')
        if idx >= self.n_bins_in_plane:
            raise IndexError(f'in-plane rot index too big -- {np.rad2deg(beta)} deg')
        return idx

    def get_indices_for_ee_pose(self, tf_ee: np.ndarray) -> tuple[int, int, int, int, int]:
        """
        Computes map indices corresponding to a given end-effector pose.

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.

        Returns:
            tuple[int, ...]: Indices in the feasibility map corresponding to the pose.

        Raises:
            IndexError: If tf_ee is outside the bounds of the map.
        """
        x_idx = self._get_xy_index(tf_ee[0, 3])
        y_idx = self._get_xy_index(tf_ee[1, 3])
        z_idx = self._get_z_index(tf_ee[2, 3])
        s_idx = self._get_sphere_point_index(tf_ee)
        r_idx = self._get_in_plane_rotation_index(tf_ee, s_idx)

        return x_idx, y_idx, z_idx, s_idx, r_idx


class FeasibilityMap6DConstructor(FeasibilityMap6DBase, FeasibilityMapConstructorBase):
    """
    Constructor for the 6D feasibility map. Uses a flexible but memory-inefficient data structure to support iterative map
    construction. Once construction is finished, the map should be converted into a `FeasibilityMap6D` instance for more
    efficient memory usage.

    Args:
        xy_limits (list[float]): [min, max] bounds for the x and y position dimensions.
        z_limits (list[float]): [min, max] bounds for the z (height) dimension.
        voxel_res (float): Resolution of the voxel grid. Defaults to 0.05.
        n_sphere_points (int): Number of discrete sphere points (used to discretize the approach direction).
        n_bins_in_plane (int): Number of bins used for the discretization of the EE in-plane rotation.
        sphere_point_tfs (np.ndarray): Optional array of shape (n_sphere_points, 4, 4) containing homogeneous 
            transformation matrices that define the approach directions. If None, they will be automatically generated 
            according to the method described by Zacharias et al.
        configuration_len (int): Length of the full configuration vector.
    """

    def __init__(self, xy_limits=None, z_limits=None, voxel_res=0.05, n_sphere_points=200, n_bins_in_plane=12,
                 sphere_point_tfs=None, configuration_len=7):
        super().__init__(xy_limits, z_limits, voxel_res, n_sphere_points, n_bins_in_plane, sphere_point_tfs)
        self.map = {}
        self.n_configurations = 0  # total number of stored robot configurations
        self.configuration_len = configuration_len

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
        Adds a new configuration to the feasibility map at the specified indices (if similar configuration is not stored already).

        Args:
            map_indices (tuple[int, ...]): Indices into the feasibility map.
            configuration (np.ndarray): Full configuration vector (shape: (DoF,)).
        """
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

    def finalize_map(self) -> "FeasibilityMap6D":
        """
        Finalizes the feasibility map construction and returns a more memory-efficient `FeasibilityMap6D`.

        Returns:
            FeasibilityMap6D: The finalized feasibility map instance.
        """
        # map stores slice indices for the configurations array
        map = np.full(
            shape=(self.n_bins_xy, self.n_bins_xy, self.n_bins_z, self.n_sphere_points,
                   self.n_bins_in_plane, 2),  # 2 for low and high index
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

            x, y, z, s, r = indices
            map[x, y, z, s, r, 0] = start
            map[x, y, z, s, r, 1] = end
            configurations[start:end, :] = confs

            curr_idx += confs.shape[0]

        return FeasibilityMap6D(map, configurations, self.xy_limits, self.z_limits, self.voxel_res, self.n_sphere_points,
                                self.n_bins_in_plane, self.sphere_point_tfs)

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
            'n_sphere_points': self.n_sphere_points,
            'n_bins_in_plane': self.n_bins_in_plane,
            'sphere_point_tfs': self.sphere_point_tfs,
            'configuration_len': self.configuration_len,
        }
        np.save(filename, save_dict)

    @classmethod
    def from_file(cls, filename: str) -> "FeasibilityMap6DConstructor":
        """
        Loads a `FeasibilityMap6DConstructor` instance from a ".npy" file.

        Args:
            filename (str): Path to the saved `.npy` file.

        Returns:
            FeasibilityMap6DConstructor: Loaded `FeasibilityMap6DConstructor` instance.
        """
        d = np.load(filename, allow_pickle=True).item()
        fm = cls(xy_limits=d['xy_limits'],
                 z_limits=d['z_limits'],
                 voxel_res=d['voxel_res'],
                 n_sphere_points=d['n_sphere_points'],
                 n_bins_in_plane=d['n_bins_in_plane'],
                 sphere_point_tfs=d['sphere_point_tfs']
                 )
        fm.map = d['map']
        fm.configuration_len = d['configuration_len']

        print(f'{cls.__name__} loaded from {filename}')
        return fm


class FeasibilityMap6D(FeasibilityMap6DBase, FeasibilityMapBase):
    """
    6D feasibility map.

    Implementation of this feasibility map is based on: Zacharias et al.:, The capability map: a tool to analyze robot
    arm workspaces, Int. Journal of Humanoid Robotics, 2013. However, instead of just saving the reachability value, 
    our map stores the robot's joint configurations (found during construction) that lead to the corresponding cell in the map.

    This map should be constructed using a `FeasibilityMap6DConstructor` followed by its `finalize_map()` method.

    Args:
        map (np.ndarray): 6D array of shape (n_bins_x, n_bins_y, n_bins_z, n_sphere_points, n_bins_in_plane, 2), storing index slices
            into `configurations`. Entries are set to -1 if no configurations are stored for that map cell.
        configurations (np.ndarray): Configuration stored in the map, excluding the wrist joint 
            (shape: (n_configurations, DoF - 1)).
        xy_limits (list[float]): [min, max] bounds for the x and y position dimensions.
        z_limits (list[float]): [min, max] bounds for the z (height) dimension.
        voxel_res (float): Resolution of the voxel grid.
        n_sphere_points (int): Number of discrete sphere points (used to discretize the approach direction).
        n_bins_in_plane (int): Number of bins used for the discretization of the EE in-plane rotation.
        sphere_point_tfs (np.ndarray): Optional array of shape (n_sphere_points, 4, 4) containing homogeneous 
            transformation matrices that define the approach directions. If None, they will be automatically generated 
            according to the method described by Zacharias et al.
    """

    def __init__(self, map: np.ndarray, configurations: np.ndarray, xy_limits=None, z_limits=None, voxel_res=0.05,
                 n_sphere_points=200, n_bins_in_plane=12, sphere_point_tfs=None):
        super().__init__(xy_limits, z_limits, voxel_res, n_sphere_points, n_bins_in_plane, sphere_point_tfs)
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
            'n_sphere_points': self.n_sphere_points,
            'n_bins_in_plane': self.n_bins_in_plane,
            'sphere_point_tfs': self.sphere_point_tfs
        }
        np.save(filename, save_dict)

    @classmethod
    def from_file(cls, filename: str) -> "FeasibilityMap6D":
        """
        Loads a `FeasibilityMap6D` instance from a ".npy" file.

        Args:
            filename (str): Path to the saved ".npy" file.

        Returns:
            FeasibilityMap6D: The loaded feasibility map instance.
        """
        d = np.load(filename, allow_pickle=True).item()
        fm = cls(d['map'], d['configurations'], d['xy_limits'], d['z_limits'], d['voxel_res'], d['n_sphere_points'],
                 d['n_bins_in_plane'], d['sphere_point_tfs'])
        print(f'{cls.__name__} loaded from {filename}')
        return fm

    def get_configurations(self, map_indices: tuple[int, ...]) -> np.ndarray:
        """
        Retrieves the configurations stored in the map at the specified indices.

        Args:
            map_indices (tuple[int, ...]): Indices into the feasibility map.

        Returns:
            np.ndarray: Array of stored configurations (shape: (n, DoF)),
            or None if no configurations are stored at the given indices.
        """
        low, high = self.map[map_indices]
        if low == -1:
            return None
        return self.configurations[low:high, :]

    def get_full_configurations(self, tf_ee: np.ndarray, forward_kinematics: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:
        """
        Retrieves full robot configurations corresponding to the given end-effector pose.

        In this map the, the output is same as `get_configurations`.

        Args:
            tf_ee (np.ndarray): Homogeneous transformation matrix of the end-effector pose.
            forward_kinematics (Callable): Unused argument. Present for API compatibility with base class.

        Returns:
            np.ndarray: Array of stored robot configurations (shape: (n, DoF)).
        """
        try:
            query_indices = self.get_indices_for_ee_pose(tf_ee)
            confs = self.get_configurations(query_indices)
            if confs is None:
                return np.array([])

            return confs

        except IndexError:
            return np.array([])
