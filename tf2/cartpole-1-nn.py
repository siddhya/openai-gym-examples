import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

import gym

def normalize(obs):
    obs[0] = obs[0] / 4.8
    obs[1] = obs[1] / (1 + abs(obs[1]))
    obs[2] = obs[2] / 0.418
    obs[3] = obs[3] / (1 + abs(obs[3]))
    
    return obs

def create_model(env):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_dim=env.observation_space.shape[0], activation="relu"))
    model.add(keras.layers.Dense(48, activation="relu"))
    #model.add(keras.layers.Dense(24, activation="relu"))
    model.add(keras.layers.Dense(env.action_space.n))
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(lr=0.01))
    return model

np.set_printoptions(suppress=True)
env = gym.make('CartPole-v0')
model = create_model(env)
gamma = 0.95
episodes = 200
total_reward = np.zeros(episodes)

for episode in range(episodes):
    observation = env.reset()
    observation = normalize(observation)
    action = np.random.randint(2)
    for t in range(200):
        #env.render()
        # Apply epsilon-greedy (linearly reduce the exploration)
        if np.random.rand() < (0.05 - (0.05 * (episode / episodes))):
        #if np.random.rand() < (0.05 / (episode + 1)):
            #print("random action chosen")
            action = np.random.randint(2)
        new_observation, reward, done, info = env.step(action)
        new_observation = normalize(new_observation)
        # Update model
        q = model.predict(observation.reshape(1, 4))
        target = q[0]
        if done:
            target[action] = reward
            print("Episode finished after {} timesteps".format(t+1))
            model.fit(observation.reshape(1, 4), target.reshape(1, 2), epochs=1, verbose=0)
            break
        q_new = model.predict(new_observation.reshape(1, 4))
        max_action = np.argmax(q_new[0])
        target[action] = reward + gamma * q_new[0][max_action]
        model.fit(observation.reshape(1, 4), target.reshape(1, 2), epochs=1, verbose=0)

        # Update for next iteration
        observation = new_observation
        action = np.argmax(model.predict(new_observation.reshape(1, 4))[0])
        total_reward[episode] += reward
    print("{} q {} avg. reward {}".format(episode, q_new[0], np.average(total_reward[:episode])))
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