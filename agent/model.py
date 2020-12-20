import torch
import torch.nn as nn

class DQN(nn.Module):
	def __init__(self, state_size, action_size):
		super(DQN, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(state_size, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 32),
			nn.ReLU(inplace=True),
			nn.Linear(32, 8),
			nn.ReLU(inplace=True),
			nn.Linear(8, action_size),
		)
	
	def forward(self, input):
		return self.main(input)
