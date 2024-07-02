from torch import nn
from torch.optim import SGD
import torch


class Actor(nn.Module):
    def __init__(self,num_inputs=8, num_outputs=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stack = nn.Sequential(
            nn.Linear(num_inputs, 200),
            nn.Linear(200, 100),
            nn.Linear(100, num_outputs)
        )
    
    def forward(self,x):
        return self.stack(x)


class Critic(nn.Module):
    def __init__(self,input_size=8, action_size=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stack = nn.Sequential(
            nn.Linear(action_size + input_size, 200),
            nn.Linear(200, 100),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.stack(x)


class Model(nn.Module):
    def __init__(self,num_inputs=8, num_outputs=2,tau=0.001, gamma=0.95, device="cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.tau = torch.tensor(tau).to(device)
        self.gamma = torch.tensor(gamma).to(device)

        self.actor = Actor(num_inputs=num_inputs, num_outputs=num_outputs)
        self.actor.to(device)
        self.critic = Critic()
        self.critic.to(device)

        self.actor_target = Actor(num_inputs=num_inputs, num_outputs=num_outputs)
        self.actor.to(device)
        self.critic_target = Critic()
        self.critic.to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.loss = nn.MSELoss()

        self.A_optim = SGD(self.actor.parameters() ,lr=0.001)
        self.C_optim = SGD(self.critic.parameters(),lr=0.001)

    def soft_update(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_critic(self, action, state, next_state, reward, done=False):
        self.C_optim.zero_grad()
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + (1 - done) * self.gamma * self.critic_target(next_state, next_action)
        critic_input = torch.cat(action, state)
        out = self.critic(critic_input)
        loss = self.loss(out, target_q)
        loss.backward()
        self.C_optim.step()

    def train_actor(self, state):
        loss = -self.critic(state, self.actor(state))
        self.A_optim.zero_grad()
        loss.backward()
        self.A_optim.step()
