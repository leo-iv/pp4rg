from typing import Callable
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from ..spaces import ConfigurationSpace


class RobotConfigDataset(Dataset):
    """
    Dataset containing robot configurations.

    Args:
        dataset_filename (str): File produced by `create_dataset`.
        cs (ConfigurationSpace): The configuration space of the robot.
        do_normalize (bool): If set to True, normalizes the dataset values to [-1,1] interval. Call `denormalize` to obtain
            the original robot configuration.
    """

    def __init__(self, dataset_filename: str, cs: ConfigurationSpace, do_normalize: bool = True):
        self.data = np.load(dataset_filename, allow_pickle=True)

        if do_normalize:
            self.data = normalize(self.data, cs)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx, :]
        return x, x  # (input, target) are same for autoencoder


def normalize(x: np.ndarray, cs: ConfigurationSpace) -> np.ndarray:
    """
    Normalizes joint configurations to [-1, 1].

    Args:
        x (np.ndarray): Shape (n, DoF) or (DoF,)
        cs (ConfigurationSpace): The configuration space of the robot.

    Returns:
        np.ndarray: Normalized configuration in [-1, 1]
    """
    lower_limits = cs.limits[:, 0]
    higher_limits = cs.limits[:, 1]
    return -1. + (x - lower_limits) * 2. / (higher_limits - lower_limits)


def denormalize(x: np.ndarray, cs: ConfigurationSpace) -> np.ndarray:
    """
    Converts normalized joint configurations in [-1, 1] back to real joint values.

    Args:
        x (np.ndarray): Shape (batch_size, DoF) or (DoF,)
        cs (ConfigurationSpace): The configuration space of the robot.

    Returns:
        np.ndarray: Denormalized configurations in original joint limits.
    """
    lower_limits = cs.limits[:, 0]
    higher_limits = cs.limits[:, 1]
    return lower_limits + (x + 1.) * (higher_limits - lower_limits) / 2


def create_dataset(filename: str, n_samples: int, cs: ConfigurationSpace, self_collision_checker: Callable[[np.ndarray], bool]):
    """
    Generates a dataset for training an autoencoder network by randomly sampling robot configurations.
    Only configurations that are free of self-collisions are saved.

    Args:
        filename (str): The name of the output file. Should end with '.npy'. The file will contain a numpy array 
            where each row represents a valid (collision-free) robot configuration.
        n_samples (int): Number of configurations to sample. 
        cs (ConfigurationSpace): The configuration space of the robot, used for sampling valid configurations.
        self_collision_checker (Callable[[np.ndarray], bool]): A function that takes a configuration 
            (a numpy array of shape (n,)) and returns True if the configuration is in self-collision, 
            and False otherwise.
    """

    dataset = np.empty((n_samples, cs.dim), dtype=np.float32)
    n_valid_confs = 0
    for _ in tqdm(range(n_samples), desc='Creating dataset'):
        conf = cs.get_random_conf()
        if not self_collision_checker(conf):
            dataset[n_valid_confs, :] = conf
            n_valid_confs += 1

    valid_dataset = dataset[:n_valid_confs, :]
    print(f"Saved dataset of {n_valid_confs} self-collision-free configurations to {filename}")
    np.save(filename, valid_dataset, allow_pickle=True)
