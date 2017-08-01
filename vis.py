import argparse
from pprint import pprint

import numpy as np
from utils import load_results
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def parse_args():
    parser = argparse.ArgumentParser(
        description="Data visualization for K-armed bandit.")

    parser.add_argument("--results-file", type=str, required=True,
                        help="File to read results from")

    args = vars(parser.parse_args())

    return args


def main(args):
  results = load_results(args["results_file"])

  actions = results["actions"]
  rewards = results["rewards"]
  optimal_arms = results["optimal_arms"]

  num_runs = results["num_runs"]
  num_episodes = results["num_episodes"]

  average_rewards = np.sum(rewards, axis=0) / num_runs
  optimal_actions = np.sum(
    actions == optimal_arms.reshape(num_runs, 1), axis=0) / num_runs

  steps = np.arange(1, num_episodes+1)

  cumulative_average_rewards = np.cumsum(average_rewards) / steps
  optimal_action_prop = 100.0 * np.cumsum(optimal_actions) / steps


  fig = plt.figure(1)

  plt.subplot(211)
  plt.plot(steps, cumulative_average_rewards)
  plt.ylabel("Average Reward")
  plt.xlabel("Steps")

  plt.subplot(212)
  plt.plot(steps, optimal_action_prop)
  plt.ylabel("% Optimal Action")
  plt.xlabel("Steps")
  plt.yticks(np.arange(0, 101, 20))

  percentage_fmt = '{x:.0f}%'
  yticks = mtick.StrMethodFormatter(percentage_fmt)
  fig.gca().yaxis.set_major_formatter(yticks)

  vis_dir = None
  if vis_dir is not None:
    vis_file = None # TODO
    plt.savefig(vis_file,
                transparent=True,
                bbox_inches='tight',
                pad_inches=0)
  else:
    plt.show()

if __name__ == "__main__":
  args = parse_args()
  main(args)
