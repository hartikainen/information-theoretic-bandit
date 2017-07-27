import numpy as np

class ArmedBanditEnv:
  """ N-armed bandit environment

  """

  def __init__(self, num_arms):
    self.num_arms = num_arms
    self.reset()

  def reset(self):
    self.bandits = [np.random.rand() for _ in range(self.num_arms)]

  def step(self, action):
    """ Step function

    Inputs:
    - action: the index of the bandit to pull

    Returns:
    - reward
    """

    if np.random.randn(1) > self.bandits[action]:
      return 1
    else:
      return -1
