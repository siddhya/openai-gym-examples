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

def normalize(obs):
    obs[0] = obs[0] / 4.8
    obs[1] = obs[1] / (1 + abs(obs[1]))
    obs[2] = obs[2] / 0.418
    obs[3] = obs[3] / (1 + abs(obs[3]))
    
    return obs

def create_model(env):
    criterion = torch.nn.MSELoss()
    model = torch.nn.Sequential(
                torch.nn.Linear(env.observation_space.shape[0], 50),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(50, 100),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(100, env.action_space.n)
            )
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    return model, criterion, optimizer

def update(model, criterion, optimizer, states, targets):
    """Update the weights of the network given a training sample. """
    target_pred = model(torch.Tensor(states))
    loss = criterion(target_pred, torch.autograd.Variable(torch.Tensor(targets)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def replay(model, target_model, criterion, optimizer, replay_buf, gamma):
    if len(replay_buf) < 20:
        return model, criterion, optimizer
    #if len(replay_buf) > 2000:
    #    replay_buf = replay_buf[-2000:]
    
    batch = random.sample(replay_buf, 20)
    states = []
    targets = []
    for obs, reward, done, action, nxt_obs in batch:
        with torch.no_grad():
            q = model(torch.Tensor(obs)).tolist()
        if done:
            q[action] = reward
        else:
            q_nxt = target_model(torch.Tensor(nxt_obs))
            #q[action] = reward + gamma * np.max(q_nxt) //Check this
            q[action] = reward + gamma * torch.max(q_nxt).item()
        states.append(obs)
        targets.append(q)
    

    update(model, criterion, optimizer, states, targets)
    return model, criterion, optimizer
                
def main():
    env = gym.make('CartPole-v1')
    model, criterion, optimizer = create_model(env)
    target_model = copy.deepcopy(model)
    gamma = 0.90
    epsilon = 0.2
    eps_decay = 0.99
    episodes = 150
    rewards = []
    replay_buf = []

    for episode in range(episodes):
        observation = env.reset()
        #observation = normalize(observation)
        #action = np.random.randint(2)
        done = False
        total_reward = 0
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
        while not done:
            #env.render()
            if random.random() < epsilon:
                #print("random action chosen")
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    #action = np.argmax(model(torch.Tensor(observation)).tolist())
                    action = torch.argmax(model(torch.Tensor(observation))).item()

            new_observation, reward, done, _ = env.step(action)
            #new_observation = normalize(new_observation)
            total_reward += reward
            replay_buf.append((observation, reward, done, action, new_observation))
            # Update model
            model, criterion, optimizer = replay(model, target_model, criterion, optimizer, replay_buf, gamma) 
            # Perpare for next step
            observation = new_observation
            #action = np.argmax(model.predict(new_observation.reshape(1, 4))[0])

        # Perpare for next episode
        rewards.append(total_reward)
        epsilon = max(epsilon * eps_decay, 0.01)
        print(termcolor.colored("{} steps {}".format(episode, total_reward), 'red'))
        # Check exit condition
        #if episode >= 100 and np.average(rewards[episode - 100:episode]) > 195:
        #    print("Success! Episodes: {} Avg Reward: {}".format(episode,
        #            np.average(rewards[max(episode - 100, 0):episode])))
        #    break

    env.close()
    #if episode + 1 == episodes:
    #    print("Fail! Episodes: {} Avg Reward: {}".format(episode + 1, np.average(rewards)))
    plot_res(rewards)

if __name__ == "__main__":
    main()