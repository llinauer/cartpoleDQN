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

The Q-functions is also called value function or state-action value function, in distinction to the value function V.





