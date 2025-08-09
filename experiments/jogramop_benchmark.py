import os
from pp4rg.benchmark import benchmark, plot_success_curves
from train_model import MODEL_FILE

TIME_LIMIT = 60
N_RUNS = 100
ALGORITHMS = ['J+-RRT', 'FM-RRT', 'Latent-J+-RRT']
SCENARIOS = [11, 21, 31, 41, 14, 24, 34, 44]
ROOT = ['experiments', 'benchmark']
FMAP_FILENAME = os.path.join('experiments', 'data', 'fmaps', 'FM4D_1e+07.npy')

if __name__ == "__main__":
    for scenario_id in SCENARIOS:
        print(f'SCENARIO: {scenario_id}')
        folder = os.path.join(*ROOT, str(scenario_id))
        os.makedirs(folder, exist_ok=True)

        benchmarks = []
        for algorithm in ALGORITHMS:
            filename = os.path.join(folder, f'{algorithm}_{N_RUNS}_{TIME_LIMIT}')
            benchmarks.append(benchmark(scenario_id, algorithm, N_RUNS, TIME_LIMIT, FMAP_FILENAME, MODEL_FILE, filename))

        plot_success_curves(os.path.join(folder, f'{scenario_id}_success_curve'), benchmarks)
