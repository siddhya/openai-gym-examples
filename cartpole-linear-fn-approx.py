import matplotlib.pyplot as plt
import numpy as np
import gym

def normalize(obs):
    obs[0] = obs[0] / 4.8
    obs[1] = obs[1] / (1 + abs(obs[1]))
    obs[2] = obs[2] / 0.418
    obs[3] = obs[3] / (1 + abs(obs[3]))
    obs = np.append(obs, obs[0] * obs[1])
    obs = np.append(obs, obs[0] * obs[2])
    obs = np.append(obs, obs[0] * obs[3])
    obs = np.append(obs, obs[1] * obs[2])
    obs = np.append(obs, obs[1] * obs[3])
    obs = np.append(obs, obs[2] * obs[3])
    return obs

np.set_printoptions(suppress=True)
env = gym.make('CartPole-v0')
weights = np.zeros((2,10))
alpha = 0.01
gamma = 0.95
episodes = 5000
total_reward = np.zeros(episodes)

for episode in range(episodes):
    observation = env.reset()
    observation = normalize(observation)
    action = np.random.randint(2)
    for t in range(200):
        #env.render()
        # Apply epsilon-greedy (linearly reduce the exploration)
        #if np.random.rand() < (0.01 - (0.01 * (episode / episodes))):
        if np.random.rand() < (0.05 / (episode + 1)):
            print("random action chosen")
            action = np.random.randint(2)
        new_observation, reward, done, info = env.step(action)
        new_observation = normalize(new_observation)
        # Decide action
        if np.dot(weights[0], new_observation) > np.dot(weights[1], new_observation):
            new_action = 0
        else:
            new_action = 1

        # Update weights
        td_err = gamma * (weights[new_action] * new_observation) - (weights[action] * observation)
        weights[action] = weights[action] + alpha * ((reward + td_err) * observation)

        # Update for next iteration
        observation = new_observation
        action = new_action
        total_reward[episode] += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    #print("{} Weights: {}".format(episode, weights))
    # Check exit condition
    if episode >= 100 and np.average(total_reward[episode - 100:episode]) > 195:
        print("Success! Episodes: {} Avg Reward: {}".format(episode,
                np.average(total_reward[max(episode - 100, 0):episode])))
        break

env.close()
if episode + 1 == episodes:
    print("Fail! Episodes: {} Avg Reward: {}".format(episode + 1, np.average(total_reward)))
avg_reward = np.zeros(episode + 1)
for i in range(1, episode + 1):
    avg_reward[i] = np.average(total_reward[:i])
plt.bar(range(len(total_reward)), avg_reward)
plt.show()