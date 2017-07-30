import numpy as np

class Arm:
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __str__(self):
    return "Arm: mean {}, std: {}".format(self.mean, self.std)

  def pull(self):
    return np.random.normal(self.mean, self.std)

class KArmedBanditEnv:
  """N-armed bandit environment

  """

  def __init__(self, k, mean_fn, std_fn):
    """Initialize bandit with k arms

    Inputs:
      k: number of bandit arms
      mean_fn: a function that, when called without arguments, returns float
        that will be used as the mean for an arm
      std_fn: a function that, when called without arguments, returns float
        that will be used as the standard deviation for an arm
    """
    self.k = k
    self.mean_fn = mean_fn
    self.std_fn = std_fn

    self.reset()

  def __str__(self):
    return "\n".join([
      "{}: {}".format(i, str(arm))
      for i, arm in enumerate(self.arms)])

  def reset(self):
    """Reset bandit"""
    self.arms = [
      Arm(self.mean_fn(), self.std_fn())
      for _ in range(self.k)
    ]

  def pull(self, a=None):
    """Environment step function

    Inputs:
    - action: the index of the arm to pull, defaults to a random arm

    Returns:
    - reward from the pulled arm
    """

    if a is None: a = np.random.choice(len(self.arms))

    return self.arms[a].pull()

  def step(self, *args):
    """Step function for the environment, which just calls self.pull"""
    return self.pull(*args)
