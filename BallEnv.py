import time
import mujoco
import mujoco.viewer
from CustomBackgroundManager import RandomIndoorBackgroundGenerator
import os
import numpy as np
import math
import mediapy as media
import cv2

# while True:
#     m = mujoco.MjModel.from_xml_path('./Assets/ball.xml')
#     d = mujoco.MjData(m)
#     d.qvel[:6] = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
#     with mujoco.viewer.launch_passive(m, d) as viewer:
#         # Set the camera to a higher and more zoomed-out position
#         with viewer.lock():
#             viewer.cam.azimuth = 90       # side view
#             viewer.cam.elevation = -45    # look down at an angle
#             viewer.cam.distance = 3.0     # zoom out
#             viewer.cam.lookat[:] = [0, 0, 0.5]  # center the scene around the ball

#         # Close the viewer automatically after 30 wall-seconds.
#         start = time.time()
#         while viewer.is_running() and time.time() - start < 5:
#             step_start = time.time()

#             mujoco.mj_step(m, d)

#             with viewer.lock():
#                 viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

#             viewer.sync()

#             time_until_next_step = m.opt.timestep - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)


class MujocoBallEnvironment:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        print(self.data)

    def composite_with_background(self, renderer_img, background_img, name = "background_image.png"):
        # Resize background to match MuJoCo render size
        background_resized = cv2.resize(background_img, (renderer_img.shape[1], renderer_img.shape[0]))

        # Convert MuJoCo render to RGB if needed
        if renderer_img.shape[2] == 4:
            renderer_img = renderer_img[:, :, :3]  # drop alpha

        # Optional: alpha blend (here we just paste over)
        blended = cv2.addWeighted(background_resized, 0.5, renderer_img, 0.5, 0)
        cv2.imwrite(os.path.abspath(name), blended )
        return blended
    
    def run_env(self):
        # Set initial velocity
        ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_body_id]
        camera_lookat = np.array([0, 0, 0.5])
        camera_pos = camera_lookat + np.array([3.0, 0.0, 1.5])

        direction = camera_pos - ball_pos
        direction = direction / np.linalg.norm(direction)
        speed = np.random.uniform(2.0, 4.0)
        self.data.qvel[:3] = direction * speed

        # Forward once to propagate qpos/qvel
        mujoco.mj_forward(self.model, self.data)

        # Set up camera
        cam = mujoco.MjvCamera()
        cam.azimuth = 90
        cam.elevation = -45
        cam.distance = 3.0
        cam.lookat[:] = [0, 0, 0.5]

        dataset_path = "./Images/"
        abs_dataset_path = os.path.abspath(dataset_path)
        background_manager = RandomIndoorBackgroundGenerator(dataset_path)
        backgrounds = background_manager.load_random_backgrounds(5)
        background = backgrounds[0]

        # Create renderer
        with mujoco.Renderer(self.model) as renderer:
            images = []

            # Simulate and render a few frames
            for i in range(30):  # simulate 30 timesteps
                mujoco.mj_step(self.model, self.data)
                renderer.update_scene(self.data, camera=cam)
                img = renderer.render()
                images.append(img.copy())  # save copy to avoid overwrite
                self.composite_with_background(img.copy(), background.copy(), name = f"background_image_{i}.png")

            
            

            # Make sure it's an absolute path (optional but safer)
            

            # Show the last image
            # while True:
            #     # cv2.imshow("Final Frame", images[-1])
            #     new_image = self.composite_with_background(images[-1], background)
            #     cv2.imshow("Final Frame", new_image)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the background images
    dataset_path = "./Images/"

    # Make sure it's an absolute path (optional but safer)
    abs_dataset_path = os.path.abspath(dataset_path)
    background_manager = RandomIndoorBackgroundGenerator(dataset_path)
    backgrounds = background_manager.load_random_backgrounds(5)
    # background_manager.visualize_backgrounds()

    # Initialize and run the Mujoco environment
    env = MujocoBallEnvironment('./Assets/ball.xml')
    env.run_env()