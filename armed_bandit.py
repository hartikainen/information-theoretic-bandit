import argparse
import numpy as np
from distutils.util import strtobool

from environment import ArmedBanditEnv
from agents import (
  DefaultAgent
)


def range_float_type(s):
  """Custom range float type for arg parser"""
  try:
    parts = list(map(float, s.split(",")))
    if len(parts) == 1:
      return parts
    elif len(parts) == 3:
      return np.arange(*parts)
  except:
    raise argparse.ArgumentTypeError(
      "range_float must be a string that, when split and parts then mapped to "
      "floats, can be passed to np.arange as arguments. E.g. '0,1.1,0.1'."
    )

def bool_type(x):
  return bool(strtobool(x))

AGENTS = {
  "default": DefaultAgent
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="N-armed bandit problem")

    parser.add_argument("-v", "--verbose", default=False, type=bool_type,
                        help="Verbose")

    parser.add_argument("-a", "--agent", default="default",
                        choices=AGENTS.keys())

    parser.add_argument("--num-arms", default=2, type=int,
                        help="Number of bandit arms")

    parser.add_argument("--num-episodes", default=1000, type=int,
                        help="Number of episodes")

    args = vars(parser.parse_args())
    return args

def main(args):
  env = ArmedBanditEnv(args["num_arms"])

  agent = AGENTS[args["agent"]](env)
  agent.learn()


if __name__ == "__main__":
  args = parse_args()
  main(args)
