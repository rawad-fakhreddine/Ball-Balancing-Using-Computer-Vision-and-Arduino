import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BallBalanceEnv(gym.Env):
    def __init__(self, difficulty_level=1):
        super(BallBalanceEnv, self).__init__()

        self.pos_limit = 0.1  # meters = 10 cm
        self.vel_limit = 1.0  # m/s limit
        self.difficulty_level = difficulty_level  # ⬅️ Curriculum step

        # Observation space: [x, y, vx, vy]
        self.observation_space = spaces.Box(
            low=np.array([-self.pos_limit, -self.pos_limit, -self.vel_limit, -self.vel_limit], dtype=np.float32),
            high=np.array([self.pos_limit, self.pos_limit, self.vel_limit, self.vel_limit], dtype=np.float32),
            dtype=np.float32
        )

        # Normalized action space in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.dt = 0.05  # 50 ms timestep
        self.g = 9.81
        #The agent fails (e.g., ball falls off the platform) OR it survives long enough
        self.max_steps = 500   # means the agent can interact with the environment up to 500 times before the episode ends
        self.current_step = 0 

    # The reset() method initializes the environment for a new episode.
    # It randomly places the ball near the center based on the difficulty level (curriculum learning),
    # sets velocity to zero, resets the step counter,
    # and returns the initial observation to the reinforcement learning agent.
    # This helps the agent learn to balance the ball starting from different positions.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Curriculum learning: increase spawn distance with difficulty
        max_init_offset = 0.01 + 0.01 * self.difficulty_level  # 1 cm to 10 cm gradually
        max_init_offset = min(max_init_offset, self.pos_limit)

        #Ensures each episode starts in a slightly different position to encourage robust learning
        self.x = np.random.uniform(-max_init_offset, max_init_offset)
        self.y = np.random.uniform(-max_init_offset, max_init_offset)
        self.vx = 0.0
        self.vy = 0.0
        self.current_step = 0

        obs = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)
        return obs, {}

    # The heart of a custom Gym environment in reinforcement learning.
    # This function simulates the environment's physics, applies the agent's action,
    # calculates the new state, and gives back a reward
    def step(self, action):
        theta4_x, theta4_y = np.clip(action, self.action_space.low, self.action_space.high)

        # Denormalize action [-1, 1] → [-25, 25] degrees
        theta4_x = float(theta4_x) * 25
        theta4_y = float(theta4_y) * 25

        # we compute the ball’s acceleration on each axis
        ax = self.g * np.sin(np.radians(theta4_x))
        ay = self.g * np.sin(np.radians(theta4_y))

        # Update Ball Velocity and Position
        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        # Package Observation for Agent
        obs = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)

        # Reward = penalize distance and speed
        dist_penalty = self.x**2 + self.y**2
        vel_penalty = 0.1 * (self.vx**2 + self.vy**2)
        reward = - (dist_penalty + vel_penalty)

        # Terminated = episode ends if the ball falls off the table (i.e., position exceeds the limit)
        terminated = bool(abs(self.x) > self.pos_limit or abs(self.y) > self.pos_limit)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # Return Everything to the RL Agent
        return obs, reward, terminated, truncated, {}

    def render(self):
        print(f"x: {self.x:.3f}, y: {self.y:.3f}, vx: {self.vx:.2f}, vy: {self.vy:.2f}")

    def set_difficulty(self, level):
        """Dynamically adjust curriculum difficulty level (from trainer)."""
        self.difficulty_level = level



