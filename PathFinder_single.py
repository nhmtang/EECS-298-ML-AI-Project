import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import matplotlib.pyplot as plt
import os

import gymnasium as gym
import gym_game

env = gym.make('Pathfinder', size = 10)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x
    
def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)
    normalized_returns = returns
    return normalized_returns

def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0

    policy.train()
    obs, info = env.reset()
    observation = np.concatenate((obs['agent'], obs['target']))

    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_pred = policy(observation)
        action_prob = F.softmax(action_pred, dim = -1)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        observation = np.concatenate((obs['agent'], obs['target']))

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward

    log_prob_actions = torch.cat(log_prob_actions)

    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, stepwise_returns, log_prob_actions

def forward_pass2(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0

    policy.train()
    obs, info = env.reset()
    env.render()

    observation = np.concatenate((obs['agent'], obs['target']))

    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_pred = policy(observation)
        action_prob = F.softmax(action_pred, dim = -1)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        observation = np.concatenate((obs['agent'], obs['target']))

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward

    log_prob_actions = torch.cat(log_prob_actions)

    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, stepwise_returns, log_prob_actions

def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()

    return loss

def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()

    return loss

def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main(): 
    MAX_EPOCHS = 5000
    DISCOUNT_FACTOR = 0.8
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    INPUT_DIM = 4
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n
    DROPOUT = 0.5

    episode_returns = []

    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

    LEARNING_RATE = 0.001
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)

        episode_returns.append(episode_return)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])

        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')

        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = dir_path + '\\Model.pt'
    torch.save(policy.state_dict(), filepath)


def test():
    env_test = gym.make('Pathfinder', render_mode='rgb_array', size = 10)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    env_record = gym.wrappers.RecordVideo(env=env_test, video_folder=dir_path, name_prefix="pathfinder", episode_trigger=lambda x: x % 2 == 0)

    MAX_EPOCHS = 5000
    DISCOUNT_FACTOR = 0.8
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    INPUT_DIM = 4
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n
    DROPOUT = 0.5

    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
    
    filepath = dir_path + '\\Model.pt'
    policy.load_state_dict(torch.load(filepath, weights_only=True))
    policy.eval()
    episode_return, stepwise_returns, log_prob_actions = forward_pass2(env_record, policy, DISCOUNT_FACTOR)
    print(episode_return)
    env_record.close()

# main()
test()




