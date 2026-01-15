import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# --- PARAMÈTRES OPTIMISÉS ---
MAX_MEMORY = 100_000  # Réduit pour éviter la mémoire excessive
BATCH_SIZE = 1024     # Batch réduit
LR = 0.0005
BLOCK_SIZE = 20
SPEED_TRAINING = 100
SPEED_DEMO = 10
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

class SnakeGameAI:
    def __init__(self, training_mode=True):
        pygame.init()
        self.w = 400
        self.h = 400
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI - Mode Entraînement' if training_mode else 'Snake AI - Démo')
        self.clock = pygame.time.Clock()
        self.training_mode = training_mode
        self.reset()
    
    def reset(self):
        self.direction = [BLOCK_SIZE, 0]
        self.head = [self.w//2, self.h//2]
        self.snake = [self.head.copy(),
                     [self.head[0] - BLOCK_SIZE, self.head[1]],
                     [self.head[0] - 2*BLOCK_SIZE, self.head[1]]]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.steps_without_food = 0
        return self.get_state_info()
    
    def _place_food(self):
        for _ in range(100):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = [x, y]
            if self.food not in self.snake:
                return
        self.food = [random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                    random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE]
    
    def play_step(self, action):
        self.frame_iteration += 1
        self.steps_without_food += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        old_distance = self._distance_to_food()
        self._move(action)
        
        reward = 0
        game_over = False
        
        if self.is_collision():
            game_over = True
            reward = -15
            if not self.training_mode:
                self._update_ui()
                time.sleep(1)
            return reward, game_over, self.score
        
        new_distance = self._distance_to_food()
        
        if self.head == self.food:
            self.score += 1
            reward = 20
            self._place_food()
            self.steps_without_food = 0
            self.snake.insert(0, list(self.head))
        else:
            self.snake.insert(0, list(self.head))
            self.snake.pop()
            
            if new_distance < old_distance:
                reward = 1
            elif new_distance > old_distance:
                reward = -1
            else:
                reward = -0.5
        
        if self.steps_without_food > 50:
            reward -= 0.1 * (self.steps_without_food - 50)
        
        if self.frame_iteration > 200 * len(self.snake):
            game_over = True
            reward = -10
        
        if not self.training_mode:
            self._update_ui()
            self.clock.tick(SPEED_DEMO)
        elif self.frame_iteration % 5 == 0:
            self._update_ui()
            self.clock.tick(SPEED_TRAINING)
        
        return reward, game_over, self.score
    
    def _distance_to_food(self):
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        
        if (pt[0] >= self.w or pt[0] < 0 or 
            pt[1] >= self.h or pt[1] < 0):
            return True
        
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        self.display.fill((15, 15, 25))
        
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, (30, 30, 45), (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, (30, 30, 45), (0, y), (self.w, y), 1)
        
        for i, pt in enumerate(self.snake):
            intensity = max(100, 255 - i * 20)
            if i == 0:
                color = (0, 255, 100)
                pygame.draw.rect(self.display, color, 
                               pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, (0, 200, 80), 
                               pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE), 2)
            else:
                color = (0, intensity, 0)
                pygame.draw.rect(self.display, color, 
                               pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, (0, max(50, intensity-30), 0), 
                               pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE), 1)
        
        pygame.draw.rect(self.display, (255, 50, 50), 
                       pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (255, 150, 150), 
                       pygame.Rect(self.food[0] + 5, self.food[1] + 5, 
                                 BLOCK_SIZE//2, BLOCK_SIZE//2))
        pygame.draw.rect(self.display, (100, 70, 30), 
                       pygame.Rect(self.food[0] + BLOCK_SIZE//2 - 2, 
                                 self.food[1] - 6, 4, 8))
        
        font = pygame.font.SysFont('arial', 20)
        font_small = pygame.font.SysFont('arial', 16)
        
        score_bg = pygame.Rect(5, 5, 150, 80)
        pygame.draw.rect(self.display, (0, 0, 0, 180), score_bg)
        pygame.draw.rect(self.display, (0, 150, 0), score_bg, 2)
        
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        length_text = font_small.render(f'Longueur: {len(self.snake)}', True, (200, 200, 200))
        steps_text = font_small.render(f'Steps: {self.frame_iteration}', True, (200, 200, 200))
        
        self.display.blit(score_text, [10, 10])
        self.display.blit(length_text, [10, 40])
        self.display.blit(steps_text, [10, 60])
        
        pygame.display.flip()
    
    def _move(self, action):
        directions = {
            'right': [BLOCK_SIZE, 0],
            'down': [0, BLOCK_SIZE],
            'left': [-BLOCK_SIZE, 0],
            'up': [0, -BLOCK_SIZE]
        }
        
        dir_order = ['right', 'down', 'left', 'up']
        
        current_dir = None
        for name, vector in directions.items():
            if self.direction == vector:
                current_dir = name
                break
        
        if current_dir is None:
            current_dir = 'right'
        
        current_idx = dir_order.index(current_dir)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (current_idx + 1) % 4
            new_dir = directions[dir_order[new_idx]]
        else:
            new_idx = (current_idx - 1) % 4
            new_dir = directions[dir_order[new_idx]]
        
        self.direction = new_dir
        self.head = [self.head[0] + self.direction[0], 
                    self.head[1] + self.direction[1]]
    
    def get_state_info(self):
        head = self.snake[0]
        
        points = {
            'up': [head[0], head[1] - BLOCK_SIZE],
            'down': [head[0], head[1] + BLOCK_SIZE],
            'left': [head[0] - BLOCK_SIZE, head[1]],
            'right': [head[0] + BLOCK_SIZE, head[1]],
            'up_left': [head[0] - BLOCK_SIZE, head[1] - BLOCK_SIZE],
            'up_right': [head[0] + BLOCK_SIZE, head[1] - BLOCK_SIZE],
            'down_left': [head[0] - BLOCK_SIZE, head[1] + BLOCK_SIZE],
            'down_right': [head[0] + BLOCK_SIZE, head[1] + BLOCK_SIZE]
        }
        
        dangers = []
        for direction in ['up', 'down', 'left', 'right', 
                         'up_left', 'up_right', 'down_left', 'down_right']:
            dangers.append(int(self.is_collision(points[direction])))
        
        food_dx = (self.food[0] - head[0]) / self.w
        food_dy = (self.food[1] - head[1]) / self.h
        
        if self.direction == [BLOCK_SIZE, 0]:
            immediate_danger = int(self.is_collision(points['right']))
        elif self.direction == [-BLOCK_SIZE, 0]:
            immediate_danger = int(self.is_collision(points['left']))
        elif self.direction == [0, BLOCK_SIZE]:
            immediate_danger = int(self.is_collision(points['down']))
        else:
            immediate_danger = int(self.is_collision(points['up']))
        
        state = dangers + [
            immediate_danger,
            int(self.direction == [-BLOCK_SIZE, 0]),
            int(self.direction == [BLOCK_SIZE, 0]),
            int(self.direction == [0, -BLOCK_SIZE]),
            int(self.direction == [0, BLOCK_SIZE]),
            food_dx,
            food_dy,
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1]
        ]
        
        return np.array(state, dtype=np.float32)

class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.value_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.value_output = nn.Linear(hidden_size // 2, 1)
        self.advantage_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage_output = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        value = F.relu(self.value_hidden(x))
        value = self.value_output(value)
        
        advantage = F.relu(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class Agent:
    def __init__(self, input_size=19, hidden_size=256):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        
        self.policy_net = DuelingDQN(input_size, hidden_size, 3)
        self.target_net = DuelingDQN(input_size, hidden_size, 3)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.memory = deque(maxlen=MAX_MEMORY)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.losses = []
    
    def get_state(self, game):
        return game.get_state_info()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        mini_sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor([a.index(1) for a in actions])
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (~dones).float() * self.gamma * next_q
        
        # CORRECTION ICI : assurer que les dimensions correspondent
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
    
    def train_short_memory(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.LongTensor([[action.index(1)]])  # Double brackets pour garder la dimension
        reward_tensor = torch.FloatTensor([reward])
        
        current_q = self.policy_net(state_tensor).gather(1, action_tensor)
        
        with torch.no_grad():
            next_q = self.target_net(next_state_tensor).max(1)[0].unsqueeze(0)
            target_q = reward_tensor + (0 if done else self.gamma * next_q)
        
        # CORRECTION ICI : target_q doit avoir la même forme que current_q
        target_q = target_q.unsqueeze(1)  # Ajouter une dimension
        
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
    
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move = [0, 0, 0]
            final_move[move] = 1
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                prediction = self.policy_net(state_tensor)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1
        
        if training:
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
        return final_move
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
        }, filename)
    
    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.n_games = checkpoint['n_games']
        self.epsilon = checkpoint['epsilon']
        print(f"Modèle chargé: {filename}")

def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/snake_ai_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    total_score = 0
    record = 0
    agent = Agent(input_size=19, hidden_size=256)
    game = SnakeGameAI(training_mode=True)
    
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    print("=" * 60)
    print("ENTRAÎNEMENT DU SNAKE AI - DÉMARRAGE")
    print("=" * 60)
    print(f"Sauvegardes dans: {save_dir}")
    print("-" * 60)
    
    try:
        episode = 0
        
        while True:
            episode += 1
            state_old = agent.get_state(game)
            episode_reward = 0
            steps = 0
            
            while True:
                action = agent.get_action(state_old, training=True)
                reward, done, score = game.play_step(action)
                state_new = agent.get_state(game)
                
                agent.train_short_memory(state_old, action, reward, state_new, done)
                agent.remember(state_old, action, reward, state_new, done)
                
                state_old = state_new
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            agent.n_games += 1
            
            agent.train_long_memory()
            
            if agent.n_games % TARGET_UPDATE == 0:
                agent.update_target_network()
            
            if score > record:
                record = score
                print(f"\n🏆 NOUVEAU RECORD! Score: {score} 🏆")
                print(f"   Episode {agent.n_games}, Epsilon: {agent.epsilon:.3f}")
                agent.save_model(f"{save_dir}/model_record_{score}.pth")
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            if agent.losses:
                recent_losses = agent.losses[-min(100, len(agent.losses)):]
                plot_losses.append(np.mean(recent_losses))
            
            if agent.n_games % 10 == 0:
                loss_value = plot_losses[-1] if plot_losses else 0
                print(f'Episode {agent.n_games:4d} | '
                      f'Score: {score:2d} ({record:2d}) | '
                      f'Mean: {mean_score:.2f} | '
                      f'Epsilon: {agent.epsilon:.3f} | '
                      f'Reward: {episode_reward:.1f} | '
                      f'Loss: {loss_value:.4f}')
            
            # CORRECTION ICI : vérifier qu'on a assez de données avant de tracer
            if agent.n_games % 20 == 0 and agent.n_games >= 20:
                ax1.clear()
                ax1.plot(plot_scores, alpha=0.7, color='blue', label='Score')
                
                # Tendance lissée (uniquement si on a assez de points)
                if len(plot_scores) > 50:
                    window_size = min(50, len(plot_scores) // 4)
                    smoothed = np.convolve(plot_scores, np.ones(window_size)/window_size, mode='valid')
                    ax1.plot(range(window_size-1, len(plot_scores)), 
                            smoothed, color='red', linewidth=2, label='Tendance')
                
                ax1.set_xlabel('Épisodes')
                ax1.set_ylabel('Score')
                ax1.set_title(f'Scores (Record: {record})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.clear()
                ax2.plot(plot_mean_scores, color='green', linewidth=2)
                ax2.set_xlabel('Épisodes')
                ax2.set_ylabel('Score moyen')
                ax2.set_title(f'Score moyen: {mean_score:.2f}')
                ax2.grid(True, alpha=0.3)
                
                ax3.clear()
                if plot_losses:
                    ax3.plot(plot_losses, color='orange', linewidth=2)
                    ax3.set_xlabel('Épisodes')
                    ax3.set_ylabel('Perte')
                    ax3.set_title(f'Perte d\'entraînement')
                    ax3.grid(True, alpha=0.3)
                
                ax4.clear()
                ax4.hist(plot_scores[-min(100, len(plot_scores)):], 
                        bins=range(0, max(plot_scores[-100:])+2 if plot_scores else 1),
                        color='purple', alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Score')
                ax4.set_ylabel('Fréquence')
                ax4.set_title('Distribution des scores récents')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.pause(0.01)
            
            if agent.n_games % 100 == 0:
                agent.save_model(f"{save_dir}/model_checkpoint_{agent.n_games}.pth")
            
            game.reset()
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("ENTRAÎNEMENT INTERROMPU")
        print("=" * 60)
        print(f"Résumé final:")
        print(f"- Épisodes: {agent.n_games}")
        print(f"- Meilleur score: {record}")
        print(f"- Score moyen: {mean_score:.2f}")
        print(f"- Epsilon final: {agent.epsilon:.3f}")
        print("-" * 60)
        
        agent.save_model(f"{save_dir}/model_final.pth")
        plt.savefig(f"{save_dir}/training_plot.png")
        
        pygame.quit()
        plt.ioff()
        plt.show()

def demo_model(model_path=None):
    print("Mode DÉMONSTRATION")
    print("-" * 60)
    
    agent = Agent(input_size=19, hidden_size=256)
    
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Modèle chargé: {model_path}")
    else:
        print("Utilisation d'un modèle non entraîné")
    
    game = SnakeGameAI(training_mode=False)
    
    scores = []
    
    try:
        for game_num in range(1, 11):
            game.reset()
            state = agent.get_state(game)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.get_action(state, training=False)
                reward, done, score = game.play_step(action)
                next_state = agent.get_state(game)
                state = next_state
                total_reward += reward
            
            scores.append(score)
            print(f"Partie {game_num}: Score = {score}, Reward = {total_reward:.1f}")
        
        print("\n" + "=" * 60)
        print(f"Résultats sur 10 parties:")
        print(f"- Score moyen: {np.mean(scores):.2f}")
        print(f"- Meilleur score: {max(scores)}")
        print(f"- Pire score: {min(scores)}")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\nDémo interrompue")
    finally:
        pygame.quit()

if __name__ == '__main__':
    print("Snake AI - Version Corrigée")
    print("1. Entraîner un nouveau modèle")
    print("2. Démo avec modèle existant")
    
    try:
        choice = input("Choix (1/2): ").strip()
    except:
        choice = "1"
    
    if choice == "1":
        train()
    elif choice == "2":
        model_path = input("Chemin du modèle (laisser vide pour modèle non entraîné): ").strip()
        demo_model(model_path if model_path else None)
    else:
        print("Choix invalide. Utilisation du mode entraînement par défaut.")
        train()