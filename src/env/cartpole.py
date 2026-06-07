import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np


class PixelObservationWrapper(gym.ObservationWrapper):
    """
    Custom wrapper to replace the default vector observation
    with a rendered RGB image observation.
    """

    def __init__(self, env, height=400, width=300):
        super().__init__(env)
        self.height = height
        self.width = width

        # Override the observation space to be an image
        # Shape is (Height, Width, Channels)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

    def observation(self, obs):
        # 1. Render the environment as an RGB array
        # This requires the base env to be initialized with render_mode="rgb_array"
        frame = self.env.render()

        # 2. Resize the image to the target dimensions
        # cv2.resize expects (width, height)
        frame_resized = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

        return frame_resized


def make_jepa_cartpole_env(height=40, width=40, stack_size=3):
    # 1. Initialize base environment to render pixel arrays
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # 2. Wrap to output 400x300x3 images instead of the 4D vector state
    env = PixelObservationWrapper(env, height=height, width=width)

    # 3. Stack the last 3 frames to provide temporal context (f_{t-2}, f_{t-1}, f_t)
    # The output shape will become (3, 400, 300, 3)
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)

    return env


if __name__ == "__main__":
    env = make_jepa_cartpole_env()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    # Output: Observation shape: (3, 400, 300, 3)

    # Create a window to display the rendering
    cv2.namedWindow("JEPA CartPole Environment", cv2.WINDOW_NORMAL)

    terminated = False
    truncated = False

    while True:
        # The observation contains 3 stacked frames: (f_{t-2}, f_{t-1}, f_t)
        # To display what's happening *right now*, we just render the most recent frame (index 2)
        current_frame = obs[2]

        # Gymnasium outputs RGB arrays, but OpenCV expects BGR arrays for rendering.
        # We must convert the color space so the colors don't look wrong (e.g., blue instead of red)
        render_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        # Display the image in the window
        cv2.imshow("JEPA CartPole Environment", render_frame)

        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Wait 50 milliseconds between frames so it runs at a viewable speed (~20 FPS)
        # Also allows us to break the loop by pressing the 'q' key
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    # Clean up
    env.close()
    cv2.destroyAllWindows()
