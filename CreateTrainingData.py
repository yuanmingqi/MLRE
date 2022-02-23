import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations

def create_training_data(
		traj_file,
		env_name,
		test_size,
		skip_state,
		save_path
):
	print('INFO: Loading {} trajs......'.format(env_name))
	trajs = pickle.load(open(traj_file, 'rb'))
	num_trajs = len(trajs['states'])

	print('INFO: Creating dataset......')

	pair_set = list()
	for (i, j) in combinations(range(num_trajs), 2):
		ti_return = sum(trajs['rewards'][i][::skip_state])
		tj_return = sum(trajs['rewards'][j][::skip_state])
		if ti_return > tj_return:
			label = 0.
		else:
			label = 1.

		pair_set.append((i, j, label))

	spt_set, qry_set = train_test_split(pair_set, test_size=test_size)

	dataset = {
		'spt_set':spt_set,
		'qry_set':qry_set
	}

	try:
		os.makedirs(save_path)
	except OSError:
		pass

	file_name = save_path + '/{}_dataset.pkl'.format(env_name.lower())
	pickle.dump(dataset, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	print('INFO: Generation accomplished!')

if __name__ == '__main__':
	env_names = ['Assault', 'Breakout', 'BeamRider', 'KungFuMaster', 'Phoenix', 'SpaceInvaders']
	num_episode = 30

	for env_name in env_names:
		traj_file = './trajs/{}episodes/trajs_ppo_{}.pkl'.format(num_episode, env_name.lower())
		create_training_data(
			traj_file=traj_file,
			env_name=env_name,
			save_path='./trajs/30episodes',
			test_size=0.2,
			skip_state=3
		)
