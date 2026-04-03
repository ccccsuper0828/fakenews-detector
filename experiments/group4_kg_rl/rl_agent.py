"""
RL-based verification routing agent and meta-classifier.

Two approaches:
1. Full RL (PPO/DQN): Learn which verification tools to invoke
2. Meta-classifier: Simple MLP combining model confidence + KG score + evidence score
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import random


# =============================================================================
# Approach 1: Simple DQN for verification routing
# =============================================================================

class VerificationEnv:
    """
    Simulated environment for RL-based verification routing.

    State: [model_confidence, kg_score, web_evidence_score, steps_taken, ...]
    Actions: 0=predict_fake, 1=predict_real, 2=query_kg, 3=web_search, 4=run_nli
    Reward: +1 correct, -1 wrong, -step_cost per verification action
    """

    ACTION_PREDICT_FAKE = 0
    ACTION_PREDICT_REAL = 1
    ACTION_QUERY_KG = 2
    ACTION_WEB_SEARCH = 3
    ACTION_RUN_NLI = 4
    NUM_ACTIONS = 5

    def __init__(self, step_cost: float = 0.1, max_steps: int = 5):
        self.step_cost = step_cost
        self.max_steps = max_steps
        self.state_dim = 8  # feature vector size

    def reset(self, model_confidence: float, true_label: int,
              kg_features: Optional[Dict] = None,
              web_features: Optional[Dict] = None) -> np.ndarray:
        """Reset environment for a new article."""
        self.true_label = true_label
        self.steps = 0
        self.done = False

        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.state[0] = model_confidence
        self.state[1] = 0.0  # kg_score (unqueried)
        self.state[2] = 0.0  # web_evidence_score (unqueried)
        self.state[3] = 0.0  # nli_score (unqueried)
        self.state[4] = 0.0  # steps_taken (normalized)
        self.state[5] = 1.0 if model_confidence > 0.5 else 0.0  # model_prediction
        self.state[6] = 0.0  # kg_queried flag
        self.state[7] = 0.0  # web_queried flag

        self._kg_features = kg_features or {}
        self._web_features = web_features or {}

        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take action, return (next_state, reward, done)."""
        self.steps += 1
        reward = 0.0

        if action == self.ACTION_PREDICT_FAKE:
            prediction = 0
            correct = (prediction == self.true_label)
            reward = 1.0 if correct else -1.0
            self.done = True

        elif action == self.ACTION_PREDICT_REAL:
            prediction = 1
            correct = (prediction == self.true_label)
            reward = 1.0 if correct else -1.0
            self.done = True

        elif action == self.ACTION_QUERY_KG:
            reward = -self.step_cost
            self.state[1] = self._kg_features.get("avg_kg_score", 0.5)
            self.state[6] = 1.0

        elif action == self.ACTION_WEB_SEARCH:
            reward = -self.step_cost
            self.state[2] = self._web_features.get("entailment_score", 0.5)
            self.state[7] = 1.0

        elif action == self.ACTION_RUN_NLI:
            reward = -self.step_cost
            self.state[3] = self._web_features.get("nli_score", 0.5)

        self.state[4] = self.steps / self.max_steps

        if self.steps >= self.max_steps and not self.done:
            prediction = 1 if self.state[0] > 0.5 else 0
            correct = (prediction == self.true_label)
            reward = 0.5 if correct else -0.5
            self.done = True

        return self.state.copy(), reward, self.done


class DQNNetwork(nn.Module):
    """Simple DQN for verification routing."""

    def __init__(self, state_dim: int = 8, num_actions: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent for learning verification routing."""

    def __init__(self, state_dim=8, num_actions=5, hidden_dim=128,
                 lr=3e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.device = "cpu"
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.num_actions = num_actions

        self.policy_net = DQNNetwork(state_dim, num_actions, hidden_dim)
        self.target_net = DQNNetwork(state_dim, num_actions, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_vals = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            return q_vals.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# =============================================================================
# Approach 2: Meta-classifier (simpler alternative)
# =============================================================================

class MetaClassifier(nn.Module):
    """
    Combines text content features + KG scores to produce final prediction.
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_meta_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    num_epochs: int = 100,
    lr: float = 1e-3,
) -> MetaClassifier:
    """
    Train a meta-classifier on combined features.

    Args:
        features: (N, D) array where D includes:
            [model_confidence, kg_violation_score, kg_verified_ratio,
             web_entailment, web_contradiction, num_triples]
        labels: (N,) binary labels.
    """
    model = MetaClassifier(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X = torch.FloatTensor(features)
    y = torch.FloatTensor(labels)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
            print(f"  [MetaClassifier] Epoch {epoch+1}/{num_epochs} "
                  f"Loss={loss.item():.4f} Acc={acc.item():.4f}")

    return model
