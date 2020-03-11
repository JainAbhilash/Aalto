import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import multiprocessing as mp
from itertools import repeat
from agent import Agent, Policy
from utils import get_space_dim
from cartpole import train, test


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="How many independent training runs to conduct")
    parser.add_argument("--episode_steps", type=int, default=None,
                        help="Maximum length of an episode")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    
    return parser.parse_args(args)


def trainer(fargs):
    trainer_id, args = fargs
    print("Trainer id", trainer_id, "started")
    # Create a Gym environment
    env = gym.make(args.env)

    # Set maximum episode length
    if args.episode_steps is not None:
        env._max_episode_steps = args.episode_steps

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    training_history = train(agent, env, args.train_episodes, silent=True,
                             train_run_id=trainer_id, early_stop=False)

    print("Trainer id", trainer_id, "finished")

    return training_history


# The main function
def main(args):
    # Create a pool with cpu_count() workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Run the train function num_runs times
    results = pool.map(trainer, zip(range(args.num_runs),
                                    repeat(args, args.num_runs)))

    # Put together the results from all workers in a single dataframe
    all_results = pd.concat(results)

    # Save the dataframe to a file
    all_results.to_pickle("rewards.pkl")

    # Plot the mean learning curve, with the satandard deviation
    sns.set()
    sns.lineplot(x="episode", y="reward", data=all_results, ci="sd")

    # Plot (up to) the first 5 runs, to illustrate the variance
    n_show = min(args.num_runs, 5)
    smaller_df = all_results.loc[all_results.train_run_id < n_show]
    sns.lineplot(x="episode", y="reward", hue="train_run_id", data=smaller_df,
                 dashes=[(2,2)]*n_show, palette="Set2", style="train_run_id")
    plt.title("Training performance")
    plt.savefig("training.png")
    plt.show()


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

