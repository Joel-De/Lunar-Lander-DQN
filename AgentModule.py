from collections import namedtuple, deque
import random
import torch
import numpy as np
from Net import DQNet


class ActionResponse:
    def __init__(self, state, action, reward, nextState, end):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.end = end


class ActionBuffer:
    def __init__(self, bufferSize, batchSize):
        self.batchSize = batchSize
        self.sequenceMemory = deque(maxlen=bufferSize)

    def __len__(self):
        return len(self.sequenceMemory)

    def add(self, state, action, reward, nextState, done):
        sequence = ActionResponse(state, action, reward, nextState, done)
        self.sequenceMemory.append(sequence)

    def sample(self):
        experiences = random.sample(self.sequenceMemory, k=self.batchSize)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.nextState for e in experiences if e is not None])).float()
        ends = torch.from_numpy(np.vstack([e.end for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, ends


# %% dqn
class AgentModule:
    def __init__(self, batchSize, learningRate, gamma, memmeorySize, device, ValidationMode=False):

        self.device = torch.device(device)
        self.ValidationMode = ValidationMode
        # model
        self.net_eval = DQNet().to(self.device)
        self.net_target = DQNet().to(self.device)

        if not ValidationMode:
            self.batchSize = batchSize
            self.learn_step = 5
            self.gamma = torch.FloatTensor([gamma]).to(self.device)
            self.tau = 1e-3
            self.optimizer = torch.optim.Adam(self.net_eval.parameters(), lr=learningRate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
            self.criterion = torch.nn.MSELoss()

            # memory
            self.memory = ActionBuffer(memmeorySize, batchSize)

    def getNextAction(self, state, epsilon):

        if (random.random() < epsilon) and (not self.ValidationMode):
            action = random.choice(np.arange(4))

        else:
            state = torch.tensor(state, dtype=torch.float32).float().unsqueeze(0).to(self.device)
            self.net_eval.eval()
            with torch.no_grad():
                action_values = self.net_eval(state)
            self.net_eval.train()
            action = np.argmax(action_values.cpu().data.numpy())


        return action

    def storeData(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) >= self.batchSize:
            experiences = self.memory.sample()
            self.updateWeights(experiences)

    def getWeights(self):
        return self.net_eval

    def loadWeights(self, model):
        self.net_target.load_state_dict(model)
        self.net_eval.load_state_dict(model)

    def stepScheduler(self):
        self.scheduler.step()

    def updateWeights(self, experiences):
        states, actions, rewards, next_states, ends = experiences
        next_states = next_states.to(self.device)
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        ends = ends.to(self.device)
        actions = actions.to(self.device)

        qTarget = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        qTargetWeighted = rewards + self.gamma * qTarget * (1 - ends)
        qEval = self.net_eval(states).gather(1, actions)

        # loss backprop
        loss = self.criterion(qEval, qTargetWeighted)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.softUpdate()

    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)
