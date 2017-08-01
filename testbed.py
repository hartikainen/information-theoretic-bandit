import argparse
from pprint import pprint
from distutils.util import strtobool
from datetime import datetime

import numpy as np

import agents
from environment import KArmedBanditEnv
from utils import dump_results

ARMS_PARAMS_CHOICES = ["constant"] + [
  fn_name for fn_name in dir(np.random) if fn_name[0] != "_"
]

def bool_type(x):
  return bool(strtobool(x))

def parse_args():
    parser = argparse.ArgumentParser(
        description="K-armed bandit testbed")

    parser.add_argument("-v", "--verbose",
                        type=bool_type,
                        default=False,
                        help="Verbose")

    parser.add_argument("--num_runs",
                        type=int,
                        default=2000,
                        help="Number runs for the experiment")

    parser.add_argument("--num_arms", "-k",
                        type=int,
                        default=10,
                        help="Number arms for the bandit")

    parser.add_argument("--arms_mean",
                        type=str,
                        default="normal",
                        choices=ARMS_PARAMS_CHOICES,
                        metavar='ARMS_MEAN', # hide choices from help
                        help="""\
Distribution to select mean for each arm from. Should correspond to a
distribution in numpy.random module. Use --arms_mean_params to change the
parameters passed to the distribution. For example:
--arms_mean=constant --arms_mean_params=1 will result in bandit whose all arms
have mean of 1, and --arms_mean_params=normal --arms_mean_params=[0,1] will
result in bandit whose all arms have mean drawn from numpy.random.normal(0, 1)
                        """)

    parser.add_argument("--arms_mean_params",
                        type=float,
                        nargs="*",
                        default=[0, 1],
                        help=("params to be passed to the distribution function"
                              "specified by --arms_mean argument."))

    parser.add_argument("--arms_std",
                        type=str,
                        default="constant",
                        choices=ARMS_PARAMS_CHOICES,
                        metavar='ARMS_STD', # hide choices from help
                        help=("Distribution to select standard deviation for"
                              " each arm from. See help for --arms_mean for"
                              " more information"))

    parser.add_argument("--arms_std_params",
                        type=float,
                        nargs="*",
                        default=[1],
                        help=("params to be passed to the distribution function"
                              "specified by --arms_mean argument."))

    DEFAULT_AGENT_CLS = "DefaultAgent"
    parser.add_argument("--agent_class", "--agent_cls",
                        type=str,
                        default=DEFAULT_AGENT_CLS,
                        help=("Name of the class (defined in agents.py) to be"
                              " used as an agent. Defaults to agents.{}."
                              "".format(DEFAULT_AGENT_CLS)))

    parser.add_argument("--num-episodes", default=1000, type=int,
                        help="Number of episodes")

    parser.add_argument("--results-file", default=None, type=str,
                        help="File to write results to")

    parser.add_argument("--epsilon", default=0.01, type=float,
                        help="epsilon for epsilon-greedy exploration")

    args = vars(parser.parse_args())

    return args


def distribution_wrapper(fn_name, fn_args):
  """Returns a function to draw values from <fn_name> distribution"""

  def draw():
    if fn_name == "constant":
      return fn_args[0] if isinstance(fn_args, list) else fn_args
    else:
      fn = getattr(np.random, fn_name)
      return fn(*fn_args)

  return draw

def get_bandit(k, arms_mean, arms_mean_params, arms_std, arms_std_params):
  mean_fn = distribution_wrapper(arms_mean, arms_mean_params)
  std_fn = distribution_wrapper(arms_std, arms_std_params)

  return KArmedBanditEnv(k, mean_fn, std_fn)

def get_agent(env, agent_cls_name, num_episodes, epsilon):
  AgentClass = getattr(agents, agent_cls_name)
  agent = AgentClass(env, num_episodes, epsilon)

  return agent

def main(args):
  k = args["num_arms"]
  agent_cls_name = args["agent_class"]

  env = get_bandit(k, args["arms_mean"], args["arms_mean_params"],
                   args["arms_std"], args["arms_std_params"])

  num_runs = args["num_runs"]
  num_episodes = args["num_episodes"]
  epsilon = args["epsilon"]

  actions = np.zeros((num_runs, num_episodes), dtype=np.min_scalar_type(k))
  rewards = np.zeros((num_runs, num_episodes), dtype=np.float32)
  optimal_arms = np.zeros(num_runs, dtype=np.min_scalar_type(k))

  for run in range(num_runs):
    env.reset()
    agent = get_agent(env, agent_cls_name, num_episodes, epsilon)
    path = agent.learn()
    tuple_path = [(p["action"], p["reward"]) for p in path]

    actions[run, :], rewards[run, :] = list(zip(*tuple_path))
    optimal_arms[run] = env.get_optimal_arm()

  results = args.copy()
  results["timestamp"] = datetime.now().isoformat()
  results["actions"] = actions
  results["rewards"] = rewards
  results["optimal_arms"] = optimal_arms
  results["epsilon"] = epsilon

  results_file = args.get("results_file")
  if results_file is not None:
    dump_results(results_file, results, file_format="pickle")

if __name__ == "__main__":
  args = parse_args()
  main(args)
