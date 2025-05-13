import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Labirent Boyutları
GRID_SIZE = 10
ACTION_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE

# Ortam Tanımı
class MazeEnv:
    def __init__(self, num_walls=20):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.num_walls = num_walls
        self.walls = set()
        self.start = None
        self.goal = None
        self.position = None
        self.reset()

    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) not in self.walls

    def reset(self):
        def is_reachable(start, goal, walls):
            from collections import deque
            visited = set()
            queue = deque([start])
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            while queue:
                x, y = queue.popleft()
                if (x, y) == goal:
                    return True
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        if (nx, ny) not in walls and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            return False

        while True:
            self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
            self.walls = set()
            while len(self.walls) < self.num_walls:
                i, j = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                self.walls.add((i, j))

            while True:
                self.start = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
                if self.start not in self.walls:
                    break

            while True:
                self.goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
                if self.goal != self.start and self.goal not in self.walls:
                    break

            if is_reachable(self.start, self.goal, self.walls):
                break

        self.position = self.start
        return self.get_state()

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE))
        state[self.position] = 1
        return state.flatten()

    def step(self, action):
        x, y = self.position
        if action == 0:
            new_pos = (x - 1, y)
        elif action == 1:
            new_pos = (x + 1, y)
        elif action == 2:
            new_pos = (x, y + 1)
        elif action == 3:
            new_pos = (x, y - 1)

        if self.is_valid_position(new_pos):
            self.position = new_pos

        reward = -1
        if self.position == self.goal:
            reward = 10

        done = self.position == self.goal
        return self.get_state(), reward, done

    def get_agent_pos(self):
        return self.position

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Eğitim Fonksiyonları
@torch.no_grad()
def update_target_model(model, target_model):
    target_model.load_state_dict(model.state_dict())

def train_agent(model, target_model, replay_buffer, optimizer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = model(states).gather(1, actions)
    next_q_values = target_model(next_states).max(1)[0].detach().unsqueeze(1)
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def epsilon_greedy(state, model, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)
    else:
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

def setup_model(state_size, action_size):
    model = QNetwork(state_size, action_size)
    target_model = QNetwork(state_size, action_size)
    update_target_model(model, target_model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, target_model, optimizer

def train():
    env = MazeEnv()
    model, target_model, optimizer = setup_model(STATE_SIZE, ACTION_SIZE)
    replay_buffer = ReplayBuffer(10000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    num_episodes = 1000
    target_update_freq = 10

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        fig, ax = plt.subplots()
        plt.title(f"Epizod {episode + 1}")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = 'white'
                if (i, j) in env.walls:
                    color = 'black'
                elif (i, j) == env.start:
                    color = 'blue'
                elif (i, j) == env.goal:
                    color = 'green'
                ax.add_patch(patches.Rectangle((j, GRID_SIZE - 1 - i), 1, 1, edgecolor='gray', facecolor=color))
        agent_patch = patches.Circle((env.position[1] + 0.5, GRID_SIZE - 1 - env.position[0] + 0.5), 0.3, color='red')
        ax.add_patch(agent_patch)
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.ion()
        plt.show()

        while not done:
            action = epsilon_greedy(state, model, epsilon)
            next_state, reward, done = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))
            train_agent(model, target_model, replay_buffer, optimizer, batch_size, gamma)
            state = next_state
            total_reward += reward

            pos = env.get_agent_pos()
            agent_patch.center = (pos[1] + 0.5, GRID_SIZE - 1 - pos[0] + 0.5)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.05)

        if episode % target_update_freq == 0:
            update_target_model(model, target_model)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Epizod {episode + 1}: Toplam ödül = {total_reward}")
        plt.ioff()
        plt.close(fig)

    print("Eğitim tamamlandı!")
    visualize_episode(env, model, "Son")

def visualize_episode(env, model, episode_num, delay=0.1):
    state = env.reset()
    done = False
    pos = env.get_agent_pos()
    path = [pos]

    fig, ax = plt.subplots()
    plt.title(f"Epizod {episode_num}")
    plt.ion()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = 'white'
            if (i, j) in env.walls:
                color = 'black'
            elif (i, j) == env.start:
                color = 'blue'
            elif (i, j) == env.goal:
                color = 'green'
            ax.add_patch(patches.Rectangle((j, GRID_SIZE - 1 - i), 1, 1, edgecolor='gray', facecolor=color))

    agent_patch = patches.Circle((pos[1] + 0.5, GRID_SIZE - 1 - pos[0] + 0.5), 0.3, color='red')
    ax.add_patch(agent_patch)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')

    while not done:
        action = epsilon_greedy(state, model, epsilon=0)
        state, _, done = env.step(action)
        pos = env.get_agent_pos()
        path.append(pos)
        agent_patch.center = (pos[1] + 0.5, GRID_SIZE - 1 - pos[0] + 0.5)
        plt.pause(delay)

    for i in range(len(path) - 1):
        y1, x1 = path[i]
        y2, x2 = path[i + 1]
        ax.plot([x1 + 0.5, x2 + 0.5], [GRID_SIZE - y1 - 0.5, GRID_SIZE - y2 - 0.5], color='blue', linewidth=2)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()
