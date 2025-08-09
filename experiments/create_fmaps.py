import numpy as np
import os

from pp4rg.fmap import *
from pp4rg import *
from pp4rg.benchmark import *

# creates just the most memory efficient FM4D
FMAPS = [FeasibilityMap4DConstructor()]
FMAP_NAMES = ['FM4D']

# uncomment following lines to create all fmap versions for comparison:
# FMAPS = [FeasibilityMap6DConstructor(), FeasibilityMap5DConstructor(), FeasibilityMap4DConstructor()]
# FMAP_NAMES = ['FM6D', 'FM5D', 'FM4D']

OUT_FOLDER = os.path.join('experiments', 'data', 'fmaps')
NUM_SAMPLES = int(1e7)
SAVE_EVERY = int(1e6)
N_SAVE_POINTS = int(NUM_SAMPLES // SAVE_EVERY)

# Initial base pose is lifted slightly off the ground to avoid false floor collisions
CONSTRUCTION_POSE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.05],
    [0.0, 0.0, 0.0, 1.0],
])

if __name__ == '__main__':
    os.makedirs(OUT_FOLDER, exist_ok=True)

    s = Scenario(0, robot_pose=CONSTRUCTION_POSE, with_platform=False)
    robot, sim = s.get_robot_and_sim(False)

    def collision_checker(conf):
        robot.reset_arm_joints(conf)
        return robot.in_collision() or robot.in_self_collision()

    def fk(conf):
        pos, orn = robot.forward_kinematics(conf)
        return np.hstack((pos, orn))

    for fmap, name in zip(FMAPS, FMAP_NAMES):
        print("----- Creating", name, "-----")

        sampler = JointSpaceSampler(fmap, robot.arm_joint_limits(), fk, collision_checker)
        filename = os.path.join(OUT_FOLDER, name)

        # saving empty fmap
        fmap.to_file(f"{filename}_constructor_{0:.0e}.npy")
        fmap.finalize_map().to_file(f"{filename}_{0:.0e}.npy")

        for i in range(N_SAVE_POINTS):
            sampler.sample(SAVE_EVERY) # sampling new configurations
            fmap.to_file(f"{filename}_constructor_{((i + 1) * SAVE_EVERY):.0e}.npy")
            print(f"Saving feasibility map to: {filename}_{((i + 1) * SAVE_EVERY):.0e}.npy")
            fmap.finalize_map().to_file(f"{filename}_{((i + 1) * SAVE_EVERY):.0e}.npy")
