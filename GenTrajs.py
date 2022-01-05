import argparse
import os
import pickle

import numpy as np
import torch

from TrajEnv import make_vec_envs

parser = argparse.ArgumentParser()
parser.add_argument(
	'--env-name', type=str
)
parser.add_argument(
	'--load-dir', type=str
)
parser.add_argument(
	'--num-episode',
	type=int,
	default=10,
	help='number of episodes, (default: 53)')
parser.add_argument(
	'--non-det',
	action='store_true',
	default=False,
	help='whether to use a non-deterministic policy')
parser.add_argument(
	'--render',
	action='store_true',
	default=False,
	help='whether to render the environment')
args = parser.parse_args()

args.det = not args.non_det

def gen_one_traj(actor_critic):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	env = make_vec_envs(
		args.env_name,
		np.random.randint(1000),
		1,
		None,
		None,
		device=device,
		allow_early_resets=False
	)

	masks = torch.zeros(1, 1)
	recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)

	done = False
	eps_states = []
	eps_actions = []
	eps_rewards = []

	steps = 0
	reward = 0
	eps_return = 0
	eps_length = 0

	obs = env.reset()

	while True:
		with torch.no_grad():
			value, action, _, recurrent_hidden_states = actor_critic.act(
				obs, recurrent_hidden_states, masks, deterministic=args.det)

		# Obser reward and next obs
		obs, reward, done, info = env.step(action)

		eps_states.append(obs.cpu().numpy()[0])
		eps_actions.append(action[0][0].cpu().numpy())
		eps_rewards.append(reward[0][0].cpu().numpy())

		steps += 1
		eps_length += 1
		eps_return += reward[0][0].cpu().numpy()

		masks.fill_(0.0 if done else 1.0)

		if steps % 1000 == 0:
			print('INFO: steps', steps)
		if done:
			# print('info: ', info)
			break

	env.close()

	eps_states = np.array(eps_states)
	eps_actions = np.array(eps_actions)
	eps_rewards = np.array(eps_rewards)

	return eps_states, eps_actions, eps_rewards, eps_return, eps_length

def save_traj():
	atari_str = 'NoFrameskip'
	if atari_str in args.env_name:
		simple_env_name = args.env_name.split('-')[0].replace(atari_str, '').lower()
	else:
		simple_env_name = args.env_name.split('-')[0].lower()

	model_dir = os.path.join(args.load_dir, args.env_name + '.pt')
	actor_critic, ob_rms = torch.load(model_dir)

	lengths = []
	rewards = []
	returns = []
	states = []
	actions = []

	for i in range(args.num_episode):
		print('INFO: {}, Episode {} =============='.format(simple_env_name, i))
		eps_states, eps_actions, eps_rewards, eps_return, eps_length = \
			gen_one_traj(actor_critic)

		print('INFO: Episode length {}, Episode return {}'.format(eps_length, sum(eps_rewards)))

		states.append(eps_states)
		actions.append(eps_actions)
		rewards.append(eps_rewards)
		returns.append(eps_return)
		lengths.append(eps_length)

	traj = {}
	traj['states'] = states
	traj['rewards'] = rewards

	save_path = 'trajs/{}episodes'.format(args.num_episode)
	try:
		os.makedirs(save_path)
	except OSError:
		pass
	file_name = '{}/trajs_ppo_{}.pkl'.format(save_path, simple_env_name)

	pickle.dump(traj, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	save_traj()
	print('INFO: Saving file.....')
	print('INFO: Trajectories generated!')