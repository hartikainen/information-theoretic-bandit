import numpy as np

class DefaultAgent:
  def __init__(self, env, num_episodes, epsilon=0.01):
    self.env = env
    self.num_episodes = num_episodes
    self.epsilon = epsilon
    self.reset()

  def reset(self):
    k = len(self.env.arms)
    self.weights = [np.random.rand() for _ in range(k)]

  def learn(self):
    k = len(self.weights)
    path = []

    for episode in range(1, self.num_episodes+1):
      if np.random.rand() < self.epsilon:
        action = np.random.randint(k)
      else:
        action = int(np.argmax(self.weights))

      reward = self.env.step(action)

      path.append({
        "action": action,
        "reward": reward
      })

    return path
