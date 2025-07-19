from ball_balance_env import BallBalanceEnv
from stable_baselines3 import PPO

total_curriculum_levels = 5
steps_per_level = 500_000

model = None

for level in range(1, total_curriculum_levels + 1):
    print(f"\n--- Training on curriculum level {level} ---")
    env = BallBalanceEnv(difficulty_level=level)
    
    if model is None:
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        model.set_env(env)

    model.learn(total_timesteps=steps_per_level)

model.save("theta2_agent_v4")
