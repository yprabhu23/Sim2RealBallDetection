import time
import mujoco
import mujoco.viewer
from CustomBackgroundManager import RandomIndoorBackgroundGenerator
import os
import numpy as np
import math
import mediapy as media
import cv2
import camera 
from scipy.spatial.transform import Rotation as R
import random
import uuid



class MujocoBallEnvironment:
    def __init__(self, model_path, num_backgrounds=10000):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.backgrounds = RandomIndoorBackgroundGenerator("./Images/").load_random_backgrounds(num_backgrounds)
        

    def run_env(self, save_dir, n=1000):
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        cam_id = self.model.camera("realsense").id
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "textured_object")
        light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "unnamed_light_0")

        os.makedirs(f"{save_dir}/mujoco_dataset", exist_ok=True)
        os.makedirs(f"{save_dir}/mujoco_no_background_dataset", exist_ok=True)

        for _ in range(n):
            mujoco.mj_resetData(self.model, self.data)

            # Randomize light
            self.data.light_xpos[light_id] = np.random.uniform([-2, -2, 1], [2, 2, 5])
            self.data.light_xdir[light_id] = np.random.uniform([-1, -1, -1], [1, 1, -0.1])
            self.model.light_directional[light_id] = 1
            self.model.light_castshadow[light_id] = 1

            # Randomize ball position
            x = np.random.normal(0, 0.25)
            y = np.random.uniform(0, 0.25)
            self.data.qpos[self.model.jnt_qposadr[self.model.body_jntadr[body_id]]:
                        self.model.jnt_qposadr[self.model.body_jntadr[body_id]] + 3] = [x, y, 10.0]

            # Randomize ball velocity
            self.data.qvel[:3] = np.random.uniform([-1, -1, -1], [1, 1, 1])
            self.data.qvel[3:6] = np.random.uniform(-1, 1, size=3)

            while True:
                mujoco.mj_step(self.model, self.data)
                renderer.update_scene(self.data, camera=cam_id)

                # RGB render
                image = renderer.render()

                # Segmentation render
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render()
                renderer.disable_segmentation_rendering()

                mask = (segmentation[..., 0] == 0)
                mask = np.expand_dims(mask, axis=-1)

                random_bg = random.choice(self.backgrounds)
                rand_bg_resized = cv2.resize(random_bg, (640, 480))
                if rand_bg_resized.ndim == 2:
                    rand_bg_resized = cv2.cvtColor(rand_bg_resized, cv2.COLOR_GRAY2RGB)
                elif rand_bg_resized.shape[2] == 4:
                    rand_bg_resized = rand_bg_resized[:, :, :3]
                composited_image = np.where(mask, image, rand_bg_resized)

                # Generate unique filename
                unique_id = uuid.uuid4()
                raw_path = f"{save_dir}/mujoco_no_background_dataset/mujoco_frame_{unique_id}.png"
                composite_path = f"{save_dir}/mujoco_dataset/mujoco_frame_{unique_id}.png"

                # Save both
                cv2.imwrite(raw_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(composite_path, cv2.cvtColor(composited_image, cv2.COLOR_RGB2BGR))

                # Stop when the ball hits the ground
                ball_center_world = self.data.xpos[body_id]
                if ball_center_world[2] <= 0:
                    break





if __name__ == "__main__":
    env = MujocoBallEnvironment('./tmp.xml')
    save_dir = "./mujoco_data_all"
    env.run_env(save_dir, n = 200)