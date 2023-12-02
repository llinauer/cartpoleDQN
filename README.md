# cartpoleDQN

This repository is a little showcase of how to use the DQN-technique from Reinforcement Learning
in a classic control problem, namely, cartpole.

I created this repository because I was reading Maxim Lapan's book "Deep Reinforcement Learning Hands-on"
at the time and wanted to apply what I've learned to a different problem than explained in the book.

This is the first part of a small series of projects I intend to do on Reinforcement Learning techniques.
Each repo will be dedicated to a different technique, applied on different benchmark of gymnasium.
Additionally, I will talk a little about how these techniques work, what is the theory behind, how you can improve performance
and so on. 
I hope you'll enjoy it.

# Deep Q-Learning

Deep Q-learning is a term, first introduced in the 2013 DeepMind paper
**Playing Atari with Deep Reinforcement Learning** (https://arxiv.org/pdf/1312.5602.pdf).
In this paper, the DeepMind team first achieved to create a Reinforcement Learning (RL) agent, that could
outperform humans in cognitively and motorically challenging games such as Breakout or Pong.
Deep Q-learning is a form of Q-Learning, a RL-technique that has been around since 1989, which uses Neural Networks (NN)
to represent a Q-function.
The term DQN, which stands for Deep Q-network was also introduced in the DeepMind paper and is used
to denote a convolutional NN, specifically designed for Deep Q-learning.
DQN and Deep Q-learning are often used interchangeably and I will stick to that.
However, keep in mind that technically, DQN has a much narrower definition than Deep Q-learning.

Ok, so DQN is concerned with a Q-function. But what is a Q-function?

## Q-functions

A Q-functions is a function, that maps a state-action pair to some real number, called the Q-value:

$ Q: s, a \rarrow \mathbb{R} $

The Q-functions is also called value function or state-action value function, in distinction to the state-value function V.
The output of Q is called the Q-value or state-action value.
In every state, for every possible action, the Q-function tells us the value of this action.
It can be defined recursively by the Bellmann equation:

$ Q(s, a) = r(s, a) + \gamma * max_{a'} Q(s', a') $

The state-action value of the current pair (s, a) is equal to the reward obtained by doing
action a in state s + the discounted state-action value of the state s', reached by performing
the action a', that leads to the state with the highest value.
This may sound a little clumsy, but it simply means that the state-action value of the current state-action pair
(s, a) is composed of the reward r(s, a) and the state-action value obtained by choosing
the optimal action a' in state s.
The discount factor is a number between 0 and 1 (usually close to 1, e.g. 0.98) that forces the agent
to put more emphasis on states that are in the near-future than those that are in the far-future.

So far so good, but how do we actually calculate Q?
Imagine you have a grid world with 16 states and 4 possible actions in each state.
Then there are 16*4 = 64 possible state action pairs (s, a). For each of these, we could list
the Q-value in a table. 
If we now interact with the environment and sample (s, a) pairs and rewards r, we can use the Bellmann equation 
to update the Q-value for each pair.
This is perfectly fine and will work. However, what if the number of possible states and actions gets larger?
For an environment with a thousand states and 20 possible actions, we already have 20k possible (s, a) pairs.
For an environment where the states are 300x200 pixel images, the number of possible states 
is something like 255^(300 * 200) ~ 10^(150000). Multiply that by some number of action, and you 
get a huge table!
If the number of states is not discrete but continuous, then this representation is actually impossible.
So what to do?
Before we answer that question, let us take a quick look at the cartpole environment.

## Cartpole

Cartpole is one of the classic control examples. A cart is mounted on some kind of rail and has a pole
attached to it. The goal is e.g. to keep the pole upright and the only way to influence the cart is
by applying some force to the cart itself. The pole itself is not actuated and must be controlled
via the cart itself.

This system is described by four variables:

* Cart position
* Cart velocity
* Pole angle
* Pole angular velocity

![Cartpole](images/cartpole_description.jpg)

So, a state of the cartpole comprises four real numbers. That means the number of possible
states is continuous, and it is impossible to come up with a table mapping Q-values to state-action pairs.
Even if we only consider the floats, which are a subset of the real numbers that table would still be unmanageable.

What to do?

## DQN

Here, the DQN comes into play. 





