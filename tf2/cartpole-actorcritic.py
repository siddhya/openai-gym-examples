import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import collections
import termcolor
import random
import torch
import copy
import gym

np.set_printoptions(suppress=True)

def plot_res(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    #clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

def create_actor_model(env, lr):
    # criterion = torch.nn.BCEWithLogitsLoss()
    # #criterion = torch.nn.MSELoss()
    # model = torch.nn.Sequential(
    #             torch.nn.Linear(env.observation_space.shape[0], 24),
    #             torch.nn.LeakyReLU(),
    #             torch.nn.Linear(24, 12),
    #             torch.nn.LeakyReLU(),
    #             torch.nn.Linear(12, env.action_space.n),
    #             torch.nn.Softmax()
    #         )
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    # return (model, criterion, optimizer)
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=env.observation_space.shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(env.action_space.n, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
    return model

def create_critic_model(env, lr):
    # criterion = torch.nn.MSELoss()
    # model = torch.nn.Sequential(
    #             torch.nn.Linear(env.observation_space.shape[0], 24),
    #             torch.nn.LeakyReLU(),
    #             torch.nn.Linear(24, 12),
    #             torch.nn.LeakyReLU(),
    #             torch.nn.Linear(12, 1),
    #         )
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    # return (model, criterion, optimizer)
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=env.observation_space.shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(1, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
    return model


def update(model_params, states, targets):
    """Update the weights of the network given a training sample. """
    model, criterion, optimizer = model_params
    target_pred = model(torch.Tensor(states))
    loss = criterion(target_pred, torch.autograd.Variable(torch.Tensor(targets)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model_params

def one_hot_encode_action(action, n_actions):
    encoded = np.zeros(n_actions, np.float32)
    encoded[action] = 1
    return encoded

def replay(a_model, c_model, replay_buf, gamma, batch_size):
    if len(replay_buf) < batch_size:
        return a_model, c_model

    batch = random.sample(replay_buf, batch_size)
    states = []
    a_targets = []
    c_targets = []

    for obs, reward, done, action, nxt_obs in batch:
        # critic
        if done:
            v = np.array([reward])
        else:
            #with torch.no_grad():
            #   v = reward + gamma * c_model[0](torch.Tensor(nxt_obs)).item()
            v = reward + gamma * np.asscalar(np.array(c_model.predict(nxt_obs.reshape([1, 4]))))

        v_old = np.asscalar(np.array(c_model.predict(obs.reshape([1, 4]))))
        # actor
        #with torch.no_grad():
        #    q = a_model[0](torch.Tensor(obs)).tolist()
        q = a_model.predict(obs.reshape([1, 4])).flatten()
        q_actual = one_hot_encode_action(action, 2)
        q_actual = [(a - b) * (v - v_old) * 0.001 for a, b in zip(q_actual, q)]
        q = [a + b for a, b in zip(q, q_actual)]
            #q[action] = v - c_model[0](torch.Tensor(obs)).item()
        #states.append(np.array([obs]))
        #c_targets.append(np.array([v]))
        #a_targets.append(np.array([q]))
        states = np.vstack([obs.reshape([1, 4])])
        c_targets = np.vstack([v])
        a_targets = np.vstack(np.array(q).reshape([1, 2]))

    #c_model = update(c_model, states, c_targets)
    #a_model = update(a_model, states, a_targets)
    c_model.train_on_batch(states, c_targets)
    a_model.train_on_batch(states, a_targets)
    return a_model, c_model
                
def main():
    RANDOM_SEED = 6
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env = gym.make('CartPole-v1')
    env.seed(RANDOM_SEED)
    gamma = 0.7
    batch_size = 20
    episodes = 300
    lr = 0.001
    update_interval = 1
    max_memory = 1000
    rewards = []
    replay_buf = collections.deque(maxlen=max_memory)
    a_model = create_actor_model(env, lr)
    c_model = create_critic_model(env, lr)

    for episode in range(episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            #env.render()
            #action = torch.multinomial(a_model[0](torch.Tensor(observation)), 1).item()
            action_probs = a_model.predict(observation.reshape([1, 4])).flatten()
            action = np.random.choice(env.action_space.n, 1, p=action_probs)[0]
            new_observation, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buf.append((observation, reward, done, action, new_observation))
            # Update model
            if episode % update_interval == 0:
                a_model, c_model = replay(a_model, c_model, replay_buf, gamma, batch_size) 
            # Perpare for next step
            observation = new_observation

        # Perpare for next episode
        rewards.append(total_reward)
        print(termcolor.colored("{} steps {}".format(episode, total_reward), 'red'))

    env.close()
    plot_res(rewards)

if __name__ == "__main__":
    main()