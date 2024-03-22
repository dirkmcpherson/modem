import gym
import numpy as np

###from tf dreamerv2 code
from gym.wrappers import TimeLimit
import memory_maze
from .dmcontrol import FrameStackWrapper
from collections import deque

# class MemoryMaze(gym.Wrapper):
class MemoryMaze:
    def __init__(self, task, obs_key="image", act_key="action", size=(64, 64), seed=0, action_repeat=2):
        # 9x9, 11x11, 13x13 and 15x15 are available
        self._env = gym.make(f"MemoryMaze-{task}-v0", seed=seed)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self._gray = False
        self._action_repeat = action_repeat

        # support frame stacking
        self._num_frames = 2
        self._frames = deque([], maxlen=self._num_frames)
        self._frames = []

        # self._observation_space = gym.spaces.Box(0, 255, (6, 224, 224), dtype=np.uint8)
        self._observation_space = gym.spaces.Box(0, 255, (self._num_frames * 3, 64, 64), dtype=np.uint8)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        # return self._env.observation_space
        return self._observation_space

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        for _ in range(self._action_repeat):
            obs, reward, done, info = self._env.step(action)
            if not self._obs_is_dict:
                obs = {self._obs_key: obs}
            self._frames.append(obs["image"])
            obs["is_first"] = False
            obs["is_last"] = done
            obs["is_terminal"] = info.get("is_terminal", False)
            if done: break

        return self._stacked_obs(), reward, done, info
        # return obs, reward, done, info

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False

        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()
    
def ImageScaleWrapper(env, size):
    class ImageScaleWrapper(gym.ObservationWrapper):
        def __init__(self, env, size):
            super().__init__(env)
            self.size = size
            obs_shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(0, 255, (obs_shape[0], size, size), dtype=np.uint8)

        def observation(self, obs):
            obs["image"] = cv2.resize(obs["image"], (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            return obs

    return ImageScaleWrapper(env, size)


def make_memorymaze_env(cfg):
    env = MemoryMaze('9x9')
    env = ImageScaleWrapper(env, cfg.img_size)
    env = TimeLimit(env, max_episode_steps=1000)
    cfg.state_dim = 1
    return env

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    env = MemoryMaze("9x9")
    obs = env.reset()
    print(obs)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        if done:
            break
        # scale up the image
        obs["image"] = cv2.resize(obs["image"], (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("image", obs["image"])
        cv2.waitKey(0)
    env.close()
    # plt.imshow(obs["image"])
    # plt.show()