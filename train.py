"""
train.py

Control the Cartpole environment with the DQN-method.
"""

import argparse
import collections
import numpy as np
import gymnasium as gym

import torch
from torch import nn

from tensorboardX import SummaryWriter

from dqn import DQN
import custom_cartpole


BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 150000
GAMMA = 0.99
REPLAY_START_SIZE = 10000
LEARNING_RATE = 3E-4
SYNC_TARGET_FRAMES = 1000


def parse_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(description='Train the cartpole system with DQN')
    parser.add_argument('--task', choices=['steady', 'upswing'], type=str,
                        help='Choose the type of task, either keeping the pole steady'
                             ' or doing an upswing', required=True)
    parser.add_argument('--max-steps', help='The maximum number of steps to run the simulation',
                        type=int, default=2000)
    args = parser.parse_args()

    return args


def sample(replay_buffer, batch_size):
    """ Sample a batch of (state, action, next_state, reward) tuples from the replay buffer.
        Return the states, actions, rewards, dones and next_states as np.arrays"""

    # choose batch_size random indices from the replay buffer
    indices = np.random.choice(len(replay_buffer), batch_size, replace=False)

    states, actions, rewards, dones, next_states = zip(*[replay_buffer[idx] for idx in indices])

    # return the states, actions, next_states
    return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),\
        np.array(dones, dtype=np.uint8), np.array(next_states)


def environment_step(env, current_state, net, epsilon):
    """ Do one step in the environment.
        Choose the action either randomly or sample the best action according to the network
        depending on epsilon.
        Return the taken action, the reward, if the environment is done or truncated and the
        next state """

    # create a random number between 0 and 1, if it is > epsilon, take a random action
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        q_vals = net(torch.tensor(current_state))
        action = torch.argmax(q_vals).item()

    # do an environment step
    next_state, reward, done, trunc, _ = env.step(action)
    return action, reward, done or trunc, next_state


def calculate_loss(batch, net, target_net):
    """ Calculate the MSELoss for predicted Q values on the batch and return it """

    states, actions, rewards, dones, next_states = batch  # unpack batch

    states_tensor = torch.tensor(states)
    actions_tensor = torch.tensor(actions)
    rewards_tensor = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)
    next_states_tensor = torch.tensor(next_states)

    state_action_values = net(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

    # do not calculate gradients for Q values calculated with target_net
    with torch.no_grad():

        # get the maximum Q value for the next_state
        next_state_action_values = target_net(next_states_tensor).max(1)[0]
        next_state_action_values[done_mask] = 0.0  # Q = 0 if the episode ended
        next_state_action_values = next_state_action_values.detach()

    # we can get an estimate of the state-action value Q(s,a) by calculating:
    # Q(s,a) ~ r(s,a) + gamma * max(Q(s',a))
    expected_state_action_values = rewards_tensor + next_state_action_values * GAMMA

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def main():
    """ main function """

    # parse cl-args
    args = parse_args()

    upswing = args.task == 'upswing'

    # define the maximum mean reward needed for stopping training
    if upswing:
        mean_reward_bound = args.max_steps * 0.93
    else:
        mean_reward_bound = args.max_steps * 0.98

    # create tensorboard writer
    writer = SummaryWriter(comment=f'_{args.task}')

    # create environment
    env = gym.make('CustomCartPole-v1', render_mode='rgb_array', upswing=upswing,
                   max_episode_steps=args.max_steps)

    # create two instances of DQN, the training and the target network
    net = DQN()
    target_net = DQN()

    # create a replay buffer
    replay_buffer = collections.deque(maxlen=REPLAY_START_SIZE)

    # training loop
    frame_idx = 0
    state, _ = env.reset()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_reward = 0.
    mean_rewards = collections.deque(maxlen=100)

    while True:

        frame_idx += 1

        # decrease epsilon every step until EPSILON_FINAL is reached
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        # do one step in the environment
        action, reward, done_or_trunc, next_state = environment_step(env, state, net, epsilon)
        total_reward += reward

        # put the sampled state, reward, action and next_state in the replay buffer
        replay_buffer.append((state, action, reward, done_or_trunc, next_state))

        state = next_state

        # check if the episode has ended
        if done_or_trunc:
            mean_rewards.append(total_reward)
            mean_reward_100 = np.mean(mean_rewards)

            if mean_reward_100 >= mean_reward_bound:
                print(f'Solved in {frame_idx} steps! Mean reward: {mean_reward_100}')
                torch.save(target_net.state_dict(),
                           f'cartpole_dqn_{args.task}_mean_reward_{mean_reward_100}_weights.pth')
                break

            # at the end of each episode, write to tensorboard
            writer.add_scalar("mean_reward_100", mean_reward_100, frame_idx)
            writer.add_scalar("total_reward", total_reward, frame_idx)
            writer.add_scalar("epsilon", epsilon, frame_idx)

            # reset the environment and the total_reward
            state, _ = env.reset()
            total_reward = 0.

        if frame_idx % 10000 == 0:
            print(f'Timestep: {frame_idx}, mean reward of the last 100 episodes: {mean_reward_100}')

        # fill the buffer before training
        if len(replay_buffer) < REPLAY_START_SIZE:
            continue

        # sync the target_net weights with the net weights every SYNC_TARGET_FRAMES frames
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())

        # gradient descent
        optimizer.zero_grad()
        batch = sample(replay_buffer, BATCH_SIZE)
        loss_tensor = calculate_loss(batch, net, target_net)
        loss_tensor.backward()
        optimizer.step()

        if frame_idx % 1000 == 0:
            writer.add_scalar('loss', loss_tensor.item(), frame_idx)
            print(f'Loss: {loss_tensor.item()}')

    writer.close()


if __name__ == '__main__':
    main()
