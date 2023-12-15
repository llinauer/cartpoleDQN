"""
play.py

Run the cartpole environment with a set of trained weights
"""

import argparse
from pathlib import Path
import gymnasium as gym

import torch

from dqn import DQN
import custom_cartpole


def parse_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(description='Run the cartpole simulation with trained weights')
    parser.add_argument('--task', choices=['steady', 'upswing', 'downswing'], type=str,
                        help='Choose the type of task, either keeping the pole steady, '
                             'doing an upswing or a downswing', required=True)
    parser.add_argument('--weights-file', type=str, help='Path to the trained weights file',
                        required=True)
    parser.add_argument('--max-steps', help='The maximum number of steps to run the simulation',
                        type=int, default=2000)
    parser.add_argument('--output-path', type=str, help='Path to save the video to',
                        default='output')
    args = parser.parse_args()

    return args


def play(env, obs, net):
    """ Play an episode in the environment """

    total_reward = 0
    while True:
        # choose action
        state_action_values = net(torch.tensor(obs))
        action = state_action_values.argmax().item()

        obs, reward, done, trunc, _ = env.step(action)
        total_reward += reward

        if done or trunc:
            break
    return total_reward


def main():
    """ main function"""

    # parse args
    args = parse_args()

    # create output path
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # create the DQN and load weights
    net = DQN()
    try:
        net.load_state_dict(torch.load(args.weights_file))
    except (RuntimeError, KeyError, ValueError) as e:
        print(f'Could not load weights file {args.weights_file}! Exception: {e}')
        return

    # create environment
    env = gym.make('CustomCartPole-v1', render_mode='rgb_array', task=args.task,
                   max_episode_steps=args.max_steps)
    obs, _ = env.reset()
    env = gym.wrappers.RecordVideo(env, video_folder=output_path)

    # run the simulation
    total_reward = play(env, obs, net)
    print(f'Simulation ended. Total reward: {total_reward}')


if __name__ == '__main__':
    main()
