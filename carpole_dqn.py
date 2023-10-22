import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from collections import deque
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LITCartpoleMLPDQN(LightningModule):
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        super().__init__()
        self.env = env
        self.observation_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = gamma

        # initialize training and target network
        self.train_network = MLP(self.observation_size, self.action_size)
        self.target_network = MLP(self.observation_size, self.action_size)
        self.target_network.load_state_dict(self.train_network.state_dict())
        self.target_network.eval()
        
        self.replay_buffer = deque(maxlen=10000)
        self.learning_rate = learning_rate
        self.optimizer = Adam(self.train_network.parameters(), lr=self.learning_rate)
        
    def forward(self, state):
        return self.train_network(state)
    
    def select_action(self, state, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.train_network(state)
                return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.train_network.state_dict())

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        q_values = self.train_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.train_network.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    dqn_model = LITCartpoleMLPDQN(env)
    trainer = Trainer(max_epochs=10)

    for episode in range(10):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = dqn_model.select_action(state)
            next_state, reward, done, _ = env.step(action)

            dqn_model.remember(state, action, reward, next_state, done)
            total_reward += reward

            state = next_state

            if len(dqn_model.replay_buffer) >= dqn_model.batch_size:
                batch = np.random.choice(dqn_model.replay_buffer, dqn_model.batch_size, replace=False)
                dqn_model.optimizer.zero_grad()
                loss = dqn_model.training_step(batch, episode)
                loss.backward()
                dqn_model.optimizer.step()

            if episode % 10 == 0:
                dqn_model.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()
