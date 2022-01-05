for env_name in Assault Breakout BeamRider KungFuMaster Phoenix SpaceInvaders
do
	python main.py --env-name ${env_name}NoFrameskip-v4 --algo ppo \
		--use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
		--num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
done
