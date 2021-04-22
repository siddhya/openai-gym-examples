import torchvision.transforms as trans
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import numpy as np
import collections
import termcolor
import random
import torch
import copy
import gym

np.set_printoptions(suppress=True)
h = 66
w = 66
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device {}".format(device))

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

class SimpleCNN(torch.nn.Module):
    def __init__(self, outputs):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 11 * 11, 100)
        self.fc2 = torch.nn.Linear(100, outputs)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, state):
        x = self.conv1(state)
        x = self.relu(x)
        #Size changes from (5, 96, 96) to (5, 48, 48)
        #x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        #Size changes from (10, 48, 48) to (10, 24, 24)
        #x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        #print("{}".format(x.shape))
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (10, 24, 24) to (1, 5760)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(x.size(0), -1)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 128)
        x = self.fc1(x)
        x = self.relu(x)

        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 128) to (1, 3)
        x = self.fc2(x)
        return(x)

def update(model, criterion, optimizer, states, targets):
    """Update the weights of the network given a training sample. """
    target_pred = model(torch.cat(states))
    loss = criterion(target_pred, torch.autograd.Variable(torch.Tensor(targets).to(device)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def replay(model, target_model, criterion, optimizer, replay_buf, gamma, batch_size):
    if len(replay_buf) < batch_size:
        return model, criterion, optimizer

    batch = random.sample(replay_buf, batch_size)
    states = []
    targets = []
    for obs, reward, done, action, nxt_obs in batch:
        with torch.no_grad():
            q = model(obs).tolist()[0]
        if done:
            q[action] = reward
        else:
            q_nxt = target_model(nxt_obs)
            q[action] = reward + gamma * torch.max(q_nxt).item()
        #states.append(obs.numpy().reshape(1, w, h))
        states.append(obs)
        targets.append(q)

    update(model, criterion, optimizer, states, targets)
    return model, criterion, optimizer

def normalize(state):
    global w, h
    state = state[:w,15:15+h]
    transformation = trans.Compose([trans.ToTensor(), trans.ToPILImage(), trans.Grayscale(), trans.ToTensor()])
    state = transformation(state.copy()).to(device)
    state[state < 0.5] = 0
    state[state > 0.5] = 1
    state = state.reshape(1, 1, w, h)
    return state

def main():
    global w, h
    gamma = 0.90
    epsilon = 0.3
    eps_decay = 0.99
    episodes = 150
    lr = 0.001
    target_update = 4
    max_memory = 10000
    batch_size = 128
    # left, nochange, right, gas, brake
    actions = [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #actions = [[-0.5, 0.25, 0], [0.5, 0.25, 0], [0, 0.5, 0], [0, 0.25, 0]]

    rewards = []
    replay_buf = collections.deque(maxlen=max_memory)
    env = gym.make('CarRacing-v0')
    model = SimpleCNN(len(actions)).to(device)
    target_model = SimpleCNN(len(actions)).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    #count = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        # Wait for the zoom to end
        for _ in range(50):
            state, _, _, _ = env.step([0, 0, 0])
        state = normalize(state)
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        while not done:
            env.render()
            if random.random() < epsilon:
                #print("random action chosen")
                action = random.randint(0, len(actions) - 1)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()
                    #print("episode: {}, reward: {}, action: {}".format(episode, total_reward, action))
            
            next_state, reward, done, _ = env.step(actions[action])
            next_state = normalize(next_state)
            replay_buf.append((state, reward, done, action, next_state))

            #if count % 50 == 0:
            #    plt.imshow(next_state.numpy().reshape(w, h), cmap='gray')
            #    plt.show()
            #count += 1

            # Perpare for next step
            total_reward += reward
            state = next_state

            # Update model
            model, criterion, optimizer = replay(model, target_model, criterion, optimizer, replay_buf, gamma, batch_size) 

        # Perpare for next episode
        rewards.append(total_reward)
        epsilon = max(epsilon * eps_decay, 0.01)
        print(termcolor.colored("{} total_reward {}".format(episode, total_reward), 'red'))

    env.close()
    plot_res(rewards)

if __name__ == "__main__":
    main()