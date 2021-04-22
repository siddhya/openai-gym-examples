import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
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

def create_model(env):
    model = torch.nn.Sequential(
                torch.nn.Linear(env.observation_space.shape[0], 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, env.action_space.n),
                torch.nn.Softmax(dim=0)
            )
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    return model, optimizer

def create_bl_model(env):
    model = torch.nn.Sequential(
                torch.nn.Linear(env.observation_space.shape[0], 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            )
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    return model, optimizer

def replay(model, optimizer, bl_model, bl_optimizer, replay_buf, gamma):
    G = []
    for t in range(len(replay_buf)):
        g = 0
        for k in range(t + 1, len(replay_buf)):
            g += (gamma ** (k - t - 1)) * replay_buf[t][1]
        G.append(g)
    G = torch.Tensor(G).squeeze()
    G = G/G.max()

    state_probs = torch.Tensor([s for (s,r,aprob) in replay_buf]) 
    v = bl_model(state_probs).squeeze()
    v = v/v.max()
    d = torch.sub(G, v)
    d1 = d.detach().clone()
    v_loss = -torch.sum(v * d)
    bl_optimizer.zero_grad()
    v_loss.backward()
    bl_optimizer.step()
    
    action_probs = torch.stack([aprob for state, r, aprob in replay_buf])
    loss = -torch.sum(torch.log(action_probs) * d1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model, optimizer, bl_model, bl_optimizer
                
def main():
    env = gym.make('CartPole-v1')
    model, optimizer = create_model(env)
    bl_model, bl_optimizer = create_bl_model(env)
    gamma = 0.99
    episodes = 1000
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        replay_buf = []
        while not done:
            action_prob = model(torch.Tensor(state))
            action = np.random.choice([0, 1], p=action_prob.detach().numpy())
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buf.append((state, reward, action_prob[action]))
            state = next_state
        # Update model
        model, optimizer, bl_model, bl_optimizer = replay(model, optimizer, 
                                                            bl_model, bl_optimizer,
                                                            replay_buf, gamma)

        # Perpare for next episode
        rewards.append(total_reward)
        print(termcolor.colored("{} steps {}".format(episode, total_reward), 'red'))

    env.close()
    plot_res(rewards)

if __name__ == "__main__":
    main()