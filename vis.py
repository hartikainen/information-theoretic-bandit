import argparse
from pprint import pprint

import numpy as np
from utils import load_result
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Data visualization for K-armed bandit.")

    parser.add_argument("--result-file", type=str, required=True,
                        help="File to read results from")

    args = vars(parser.parse_args())

    return args


def main(args):
  data = load_result(args["result_file"])
  path = data["agent_path"]
  env = data["env"]

  optimal_arm = np.argmax([arm["mean"] for arm in env["arms"]])
  rewards = [p["reward"] for p in path]
  optimal_actions = [p["action"] == optimal_arm for p in path]
  steps = np.arange(1, len(rewards)+1)

  average_rewards = np.cumsum(rewards) / steps
  optimal_action_prop = np.cumsum(optimal_actions) / steps

  fig = plt.figure(1)
  plt.plot(steps, average_rewards)

  fig = plt.figure(2)
  plt.plot(steps, optimal_action_prop)

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
