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


if __name__ == "__main__":

    fx, fy = 379.004150390625, 378.7071228027344  # Focal lengths
    cx, cy = 313.1700439453125, 250.78013610839844  # Principal point

    r = R.from_quat([0.7071, 0, 0, 0.7071])  # Example quaternion
    print(r.as_matrix())
    # Camera intrinsic matrix
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    dataset_path = "./Images/"
    abs_dataset_path = os.path.abspath(dataset_path)
    background_manager = RandomIndoorBackgroundGenerator(dataset_path)
    backgrounds = background_manager.load_random_backgrounds(5)
    background = backgrounds[0]
    rand_image = background
    
    # # Camera extrinsic parameters (position and orientation)
    camera_position = np.array([0, 0, 4])  
    
    # # Create rotation matrix (camera looking down the z-axis)
    camera_rotation = np.array([
    [ 1., 0.,  0.],
    [ 0.,  0.,  -1.],
    [ 0.,  1.,  0.]])  #
    
    # Ball center in world coordinates (in front of camera)
    # ball_center_world = np.array([0, 0, 0])  # 3 meters in front of camera
    


    model = mujoco.MjModel.from_xml_path('./tmp.xml')
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "textured_object")
    mean = 0
    sigma = 0.25
    x = np.random.normal(0, sigma)
    y = np.random.uniform(0, sigma)

    data.qpos[model.jnt_qposadr[model.body_jntadr[body_id]]:model.jnt_qposadr[model.body_jntadr[body_id]]+3] = [x, y, 10.0]


    vx = np.random.uniform(-1.0, 1.0)   # random x velocity
    vy = np.random.uniform(-1.0, 1.0)   # random y velocity
    vz = 5.0                            # high upward velocity

    wx = np.random.uniform(-1.0, 1.0)   # random x angular velocity
    wy = np.random.uniform(-1.0, 1.0)   # random y angular velocity
    wz = np.random.uniform(-1.0, 1.0)   # random z angular velocity

    data.qvel[:3] = [vx, vy, 0]        # Set linear velocity
    data.qvel[3:6] = [wx, wy, wz]   
    i = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
    
        start_time = time.time()
        while time.time() - start_time < 30:
            mujoco.mj_step(model, data)
            cam_id = model.camera("realsense").id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
            
            renderer.update_scene(data, camera=cam_id)

            # Render RGB image
            image = renderer.render()

            # Render segmentation image
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            renderer.disable_segmentation_rendering()

            # Create mask where ball (segid 0) is present
            mask = (segmentation[..., 0] == 0)
            mask = np.expand_dims(mask, axis=-1)

            # Resize background only once (if static)
            if i == 0:
                rand_image_resized = cv2.resize(rand_image, (640, 480))

            # Compose image
            composited_image = np.where(mask, image, rand_image_resized)

            # Save images
            cv2.imwrite(f'./mujoco_no_background_dataset/mujoco_frame_{i}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'./mujoco_dataset/mujoco_frame_{i}.png', cv2.cvtColor(composited_image, cv2.COLOR_RGB2BGR))

            i += 1
            viewer.sync()

