from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Simple convolutional encoder for 84x84 stacked grayscale frames."""

    def __init__(self, in_channels: int, output_dim: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x


class ActorCritic(nn.Module):
    """Combined policy and value network."""

    def __init__(self, in_channels: int, num_actions: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def get_policy_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(obs)
        logits = self.policy_head(embedding)
        value = self.value_head(embedding).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.get_policy_and_value(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.get_policy_and_value(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


class QNetwork(nn.Module):
    """CNN-based Q-value estimator shared by DQN variants."""

    def __init__(self, in_channels: int, num_actions: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(obs)
        return self.q_head(embedding)
