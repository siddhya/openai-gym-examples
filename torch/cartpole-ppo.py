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
    optimizer = torch.optim.Adam(model.parameters(), 0.002)
    return model, optimizer

def replay(model, optimizer, bl_model, bl_optimizer, replay_buf, gamma):
    
    d = []
    v = None
    for state, reward, action, action_prob, next_state, done in replay_buf:
        if v is None:
            v = bl_model(torch.Tensor(state)).squeeze()
        else:
            v = v_next
        v_next = bl_model(torch.Tensor(next_state)).squeeze()
        d.append(reward + (1 - done) * v_next * gamma - v)
    
    a = []
    for i in range(len(d)):
        ai = 0
        for j in range(i, len(d)):
            ai = ai + (gamma ** (len(d) - j)) * d[j].detach().clone()
        a.append(ai)

    p_old = []
    p_new = []
    for state, _, action, action_prob, _, _ in replay_buf:
        p_new.append(torch.log(model(torch.Tensor(state))[action]))
        p_old.append(torch.log(action_prob)[action])
    
    v_loss = torch.pow(torch.stack(d), 2).mean()
    bl_optimizer.zero_grad()
    v_loss.backward()
    bl_optimizer.step()

    #ratio = torch.exp((torch.log(model(torch.Tensor(state))[action]) - torch.log(action_prob)[action]))
    ratio = torch.exp(torch.stack(p_new) - torch.stack(p_old))
    ratio = torch.clamp(ratio, 1.0 - 0.2, 1 + 0.2)
    loss = -(ratio * torch.stack(a)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
                
def main():
    env = gym.make('CartPole-v1')
    model, optimizer = create_model(env)
    bl_model, bl_optimizer = create_bl_model(env)
    gamma = 0.99
    episodes = 1000
    ppo_batch = 20
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        replay_buf = []
        while not done:
            with torch.no_grad():
                action_prob = model(torch.Tensor(state))
            action = np.random.choice([0, 1], p=action_prob.detach().numpy())
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buf.append((state, reward, action, action_prob, next_state, done))
            # Update model
            if total_reward % ppo_batch == 0 or done:
                replay(model, optimizer, bl_model, bl_optimizer, replay_buf, gamma)
                replay_buf = []
            state = next_state

        # Perpare for next episode
        rewards.append(total_reward)
        print(termcolor.colored("{} steps {}".format(episode, total_reward), 'red'))

    env.close()
    plot_res(rewards)

if __name__ == "__main__":
    main()