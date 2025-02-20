import multiprocessing
import time
import os
import csv

import popgym
from popgym.envs.battleship import Battleship

# Test FPS for MinAtar environment
# Source: https://github.com/kenjyoung/MinAtar/tree/master
import gymnasium as gym
# TestEnv = gym.make('MinAtar/Asterix-v1')

NUM_STEPS = 32

# Test FPS for popgym environment
# Source: https://github.com/proroklab/popgym
TestEnv = Battleship()

# seed = int(os.getenv("SEED"))
seeds = list(range(10))


def run_sample(e, num_steps, seed):
    env = e
    env.reset(seed=seed)
    start = time.time()
    for i in range(num_steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset(seed=seed)
    end = time.time()
    elapsed = end - start
    if elapsed == 0:
        elapsed = 1e-6
    fps = num_steps / elapsed
    # print(f"num_steps: {num_steps}, elapsed: {elapsed}, FPS: {fps:.0f}")
    return fps

def main():
    print(f"Testing environment: {TestEnv}")
    # Loop over each seed.
    for seed in seeds:
        print(f"\nTesting with seed: {seed}")
        # Run single environment test (baseline reference)
        fps_single = run_sample(TestEnv, NUM_STEPS, seed)
        print(f"{TestEnv} (1x) FPS: {fps_single:.0f} with seed: {seed}")

        for n in range(1, 10):
            num_workers = 2**n
            # Prepare the arguments for each worker
            envs = num_workers * [TestEnv]
            steps = num_workers * [int(NUM_STEPS / num_workers)]
            print(f"Running {num_workers} workers with {steps[0]} steps each")
            seed_list = num_workers * [seed]
            with multiprocessing.Pool(processes=num_workers) as p:
                fps_multi = sum(p.starmap(run_sample, zip(envs, steps, seed_list)))
            # print(f"{TestEnv} ({num_workers}x) FPS: {fps_multi:.0f} with seed: {seed}")
            csv_file = 'POPGymMinatar_fps_results.csv'
            write_header = not os.path.exists(csv_file)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(["Environment", "Num Envs", "Num Steps", "FPS", "Seed"])
                writer.writerow([TestEnv, num_workers, steps, f"{fps_multi:.0f}", seed])



if __name__ == "__main__":
    main()