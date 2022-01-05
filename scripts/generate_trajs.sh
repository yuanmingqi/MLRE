for env_name in SpaceInvaders BeamRider Breakout Phoenix KungFuMaster Assault
do
	python GenTrajs.py --env-name ${env_name}NoFrameskip-v4 \
		--load-dir experts/PPO --num-episode 30 --non-det

done
