import cloudpickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow import log_metric, log_param, log_artifact

from tensorboardX import SummaryWriter

import numpy as np
from rlenergy_gym.envs import rl_energy_env

with open('prepared_env.pkl', 'rb') as f:
    env_pack = cloudpickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the neural network.')
    parser.add_argument('--epochs', default=10000, type=int, help='training epochs')
    parser.add_argument('--model', default='ddpg', help='RL model')
    parser.add_argument('--net_model', default='nn33', help='name of the neural network')
    parser.add_argument('--hidden_dim', default=128, help='hidden dimensions in neural network')
    parser.add_argument('--value_lr', default=1e-2, help='Value network learning rate')
    parser.add_argument('--policy_lr', default=1e-2, help='Policy network learning rate')
    parser.add_argument('--gamma', default=0.95, type=float, help='discount factor')

    args = parser.parse_args()

    battery, result_df, resource = env_pack
    battery_copy = battery.copy()

    if args.model == 'ddpg':
        from models.ddpg import OUNoise, ReplayBuffer
        if args.net_model == 'nn33':
            from models.ddpg import ValueNetwork3LinearLayer as ValueNetwork
            from models.ddpg import PolicyNetwork3LinearLayer as PolicyNetwork
        elif args.net_model == 'nn44':
            from models.ddpg import ValueNetwork4LinearLayer as ValueNetwork
            from models.ddpg import PolicyNetwork4LinearLayer as PolicyNetwork


    env = rl_energy_env.EnergyEnv(battery_copy, resource, result_df)

    writer = SummaryWriter()

    ou_noise = OUNoise(env.action_space)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = args.hidden_dim

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    def ddpg_update(batch_size,
                    gamma=args.gamma,
                    min_value=-np.inf,
                    max_value=np.inf,
                    soft_tau=1e-2):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = value_net(state, policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = target_policy_net(next_state)
        target_value = target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = value_net(state, action)
        value_loss = value_criterion(value, expected_value.detach())

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    value_lr = args.value_lr
    policy_lr = args.policy_lr


    value_optimizer = optim.SGD(value_net.parameters(), lr=value_lr)
    policy_optimizer = optim.SGD(policy_net.parameters(), lr=policy_lr)

    value_criterion = nn.MSELoss()

    replay_buffer_size = 100000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    max_steps = len(resource)
    max_frames = max_steps * int(args.epochs)
    print('Max frame is', max_frames)
    frame_idx = 0
    rewards = []
    batch_size = 256

    while frame_idx < max_frames:
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = policy_net.get_action(state)
            action_with_noise = ou_noise.get_action(action, step)
            diff = action_with_noise - action
            action = action_with_noise
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                ddpg_update(batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if frame_idx % max(1000, max_steps + 1) == 0:
                writer.add_scalar('reward', rewards[-1], frame_idx)

            if done:
                break

        rewards.append(episode_reward)
