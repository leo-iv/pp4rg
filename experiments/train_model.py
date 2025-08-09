import os
import torch
import numpy as np
from torchsummary import summary

from pp4rg import *
from pp4rg.latent import *
from pp4rg.benchmark import *

FOLDER = os.path.join('experiments', 'data', 'latent')
DATASET_FILE = os.path.join(FOLDER, 'dataset.npy')
MODEL_FILE = os.path.join(FOLDER, 'model.pt')
N_DATASET_SAMPLES = 1000000

SEED = 42

if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(FOLDER, exist_ok=True)

    s = Scenario(0)
    robot, sim = s.get_robot_and_sim(with_gui=False)
    cs = ConfigurationSpace(robot.arm_joint_limits(), lambda c: False)

    def collision_checker(conf):
        robot.reset_arm_joints(conf)
        return robot.in_self_collision()

    if not os.path.exists(DATASET_FILE):
        create_dataset(DATASET_FILE, N_DATASET_SAMPLES, cs, collision_checker)

    dataset = RobotConfigDataset(DATASET_FILE, cs)
    model = Autoencoder(cs.dim)

    print("Model summary:")
    summary(model, (9,), TrainConfig.batch_size)

    train_model(model, dataset, MODEL_FILE)
