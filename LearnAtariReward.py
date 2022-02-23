import numpy as np
import torch
import pickle
import os

from torch import nn, optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from itertools import combinations


class RewardNetwork(nn.Module):
	def __init__(
			self,
			num_inputs,
			hidden_size=512
	):
		super(RewardNetwork, self).__init__()

		self.main = nn.Sequential(
			nn.Conv2d(num_inputs, 32, 8, stride=4), nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
			nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(),
			nn.Flatten(),
			nn.Linear(32 * 7 * 7, hidden_size), nn.ReLU(),
			nn.Linear(hidden_size, 1)
		)

	def compute_return(self, traj):
		estimated_rewards = self.main(traj)
		sum_rewards = torch.sum(estimated_rewards)

		# regularizer = torch.abs((len(traj) / sum_rewards) - 1.)
		regularizer = len(traj) / sum_rewards

		return sum_rewards, regularizer

	def compute_reward(self, state):
		return self.main(state)

	def forward(self, traj_i, traj_j):
		r_i, regularizer_r_i = self.compute_return(traj_i)
		r_j, regularizer_r_j = self.compute_return(traj_j)

		return torch.cat((r_i.unsqueeze(0), r_j.unsqueeze(0)), 0), regularizer_r_i + regularizer_r_j


class MLRE:
	def __init__(
			self,
			training_tasks,
			testing_task,
			meta_epoch,
			finetune_epoch,
			alpha,
			beta,
			device,
			trajs_dir,
			skip_state,
			save_dir
	):
		self.training_tasks = training_tasks
		self.testing_task = testing_task
		self.meta_epoch = meta_epoch
		self.finetune_epoch = finetune_epoch
		self.env_name = testing_task
		self.device = device

		self.meta_network = RewardNetwork(num_inputs=4)
		self.meta_network.to(device)
		self.optimizer_meta = optim.Adam(
			params=self.meta_network.parameters(),
			lr=beta
		)
		self.beta = beta

		self.task_network = RewardNetwork(num_inputs=4)
		self.task_network.to(device)
		self.optimizer_task = optim.Adam(
			params=self.task_network.parameters(),
			lr=alpha
		)

		self.save_dir = save_dir
		self.skip_state = skip_state
		self.trajs_dir = trajs_dir

	def load_dataset(self):
		all_trajs = dict()
		all_pair_sets = dict()

		for task in self.training_tasks:
			trajs_file = self.trajs_dir + '/trajs_ppo_{}.pkl'.format(task.lower())
			all_trajs[task] = pickle.load(open(trajs_file, 'rb'))
			pair_set_file = self.trajs_dir + '/{}_dataset.pkl'.format(task.lower())
			all_pair_sets[task] = pickle.load(open(pair_set_file, 'rb'))

		return all_trajs, all_pair_sets

	def meta_train(self, all_trajs, all_pair_sets):
		loss_criterion = nn.CrossEntropyLoss()

		for epoch in range(self.meta_epoch):
			for task in self.training_tasks:
				if task == self.testing_task:
					continue

				self.task_network.load_state_dict(self.meta_network.state_dict())

				spt_set = all_pair_sets[task]['spt_set']
				qry_set = all_pair_sets[task]['qry_set']
				trajs = all_trajs[task]

				for idx, (i, j, label) in enumerate(spt_set):
					traj_i = trajs['states'][i]
					traj_j = trajs['states'][j]
					label = np.array([label])
					traj_i = torch.from_numpy(traj_i[::self.skip_state]).float().to(self.device)
					traj_j = torch.from_numpy(traj_j[::self.skip_state]).float().to(self.device)
					label = torch.from_numpy(label).to(self.device)

					self.optimizer_task.zero_grad()

					outputs, regularizer = self.task_network.forward(traj_i, traj_j)
					outputs = outputs.unsqueeze(0)
					loss = loss_criterion(outputs, label.long()) + regularizer
					loss.backward()
					self.optimizer_task.step()

				qry_loss = torch.zeros(1)
				qry_loss = qry_loss.to(self.device)
				''' calculate the loss on the query set '''
				for idx, (i, j, label) in enumerate(qry_set):
					traj_i = trajs['states'][i]
					traj_j = trajs['states'][j]
					label = np.array([label])
					traj_i = torch.from_numpy(traj_i[::self.skip_state]).float().to(self.device)
					traj_j = torch.from_numpy(traj_j[::self.skip_state]).float().to(self.device)
					label = torch.from_numpy(label).to(self.device)

					outputs, regularizer = self.task_network.forward(traj_i, traj_j)
					outputs = outputs.unsqueeze(0)
					loss = loss_criterion(outputs, label.long()) + regularizer
					qry_loss += loss

				''' update the meta network '''
				self.optimizer_task.zero_grad()
				gradients = torch.autograd.grad(qry_loss, self.task_network.parameters())
				for (grad, param) in zip(gradients, self.meta_network.parameters()):
					param.data = param.data - self.beta * grad

				print('INFO: Meta Training epoch {} task {}'.format(epoch, task))

	def train(self):
		loss_criterion = nn.CrossEntropyLoss()

		print('INFO: Loading training data......')
		all_trajs, all_pair_sets = self.load_dataset()
		print('INFO: Meta Training......')
		# self.meta_train(all_trajs, all_pair_sets)

		spt_set = all_pair_sets[self.env_name]['spt_set']
		qry_set = all_pair_sets[self.env_name]['qry_set']
		trajs = all_trajs[self.env_name]

		for epoch in range(self.finetune_epoch):
			total_rank_loss = 0.
			total_actual_loss = 0.
			for idx, (i, j, label) in enumerate(spt_set):
				traj_i = trajs['states'][i]
				traj_j = trajs['states'][j]
				traj_i_tr = sum(trajs['rewards'][i][::self.skip_state])
				traj_j_tr = sum(trajs['rewards'][j][::self.skip_state])
				tr = torch.from_numpy(np.array([traj_i_tr, traj_j_tr]))
				label = np.array([label])
				traj_i = torch.from_numpy(traj_i[::self.skip_state]).float().to(self.device)
				traj_j = torch.from_numpy(traj_j[::self.skip_state]).float().to(self.device)
				label = torch.from_numpy(label).to(self.device)

				self.optimizer_meta.zero_grad()

				outputs, regularizer = self.meta_network.forward(traj_i, traj_j)
				outputs = outputs.unsqueeze(0)
				''' punishments for unnormal outputs '''
				rank_loss = loss_criterion(outputs, label.long())
				(rank_loss + regularizer).backward()
				actual_loss = torch.abs(torch.sum(outputs.detach().cpu() - tr))
				self.optimizer_meta.step()

				item_rank_loss = rank_loss.item()
				item_actual_loss = actual_loss.item()
				total_rank_loss += item_rank_loss
				total_actual_loss += item_actual_loss

			with torch.no_grad():
				total_eval_loss = 0.0
				for idx, (i, j, label) in enumerate(qry_set):
					traj_i = trajs['states'][i]
					traj_j = trajs['states'][j]
					label = np.array([label])
					traj_i = torch.from_numpy(traj_i[::self.skip_state]).float().to(self.device)
					traj_j = torch.from_numpy(traj_j[::self.skip_state]).float().to(self.device)
					label = torch.from_numpy(label).to(self.device)

					outputs, regularizer = self.meta_network.forward(traj_i, traj_j)
					outputs = outputs.unsqueeze(0)
					eval_loss = loss_criterion(outputs, label.long())
					item_loss = eval_loss.item()
					total_eval_loss += item_loss
			total_eval_loss = 0.

			print('INFO: Fine-tuning epoch {} total rank loss {:.3f} total actual loss {:.3f} eval loss {:.3f}'.format(
				epoch, total_rank_loss, total_actual_loss, total_eval_loss))

		print('INFO: Training finished! Saving model......')

		try:
			os.makedirs(self.save_dir)
		except OSError:
			pass

		torch.save(self.meta_network,
		           self.save_dir + '/{}_learned_reward.pt'.format(self.env_name.lower()))


if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
	mlre = MLRE(
		training_tasks=['Assault', 'Breakout', 'BeamRider', 'KungFuMaster', 'Phoenix', 'SpaceInvaders'],
		testing_task='SpaceInvaders',
		meta_epoch=10,
		finetune_epoch=100,
		alpha=0.0005,
		beta=0.0001,
		device=device,
		save_dir='./models',
		trajs_dir='./trajs/50episodes',
		skip_state=3
	)
	mlre.train()
