import xml.etree.ElementTree as ET
from lxml import etree
from scipy.spatial.transform import Rotation
import mujoco
import numpy as np
from collections import namedtuple
from typing import Dict, Any, Optional, Tuple


from ipdb import set_trace as bp

CameraParams = namedtuple("CameraParams", ["pos", "quat", "fovy"])


def nominal_camera_params(model, camera_body) -> CameraParams:
    camera_body = next(
        (model.camera(i) for i in range(model.ncam) if model.camera(i).name == camera_body)
    )
    return CameraParams(
        pos=camera_body.pos.copy(),
        quat=camera_body.quat.copy(),
        fovy=camera_body.fovy,
    )


def randomize_camera(model, camera_name, nominal_params):
    camera_offset = np.random.uniform(-0.05, 0.05, 3)
    model.body(camera_name).pos = nominal_params.pos + camera_offset

    camera_rotation_perturb = Rotation.from_euler(
        "XYZ", np.random.uniform(-2.5, 2.5, 3), degrees=True
    )
    model.body(camera_name).quat = (
        Rotation.from_quat(
            nominal_params.quat,
            scalar_first=True,
        )
        * camera_rotation_perturb
    ).as_quat(scalar_first=True)

    model.camera(camera_name).fovy = nominal_params.fovy + np.random.uniform(-5, 5)


def set_global_camera_xml(
    model, assets, camera_poses_in_world_frame: Optional[Dict[str, np.ndarray]]
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Add cameras specified in camera_poses_in_world_frame to the model's XML.
    Returns new model and data with only the specified cameras. This should be called before
    setting up any wrist cameras/auxillary cameras as it removes all pre-existing cameras and adds the
    configured global cameras to the scene.
    """
    xml_path = "tmp.xml"
    mujoco.mj_saveLastXML(xml_path, model)

    tree = etree.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    for camera in root.findall(".//camera"):
        parent = camera.getparent()
        if parent is not None:
            parent.remove(camera)

    # Randomize global cameras around their corresponding real world positions
    #Instead of randoming global camera, fix the quat and the position. Or, set the camera poses in world frame to be none
    # if camera_poses_in_world_frame is not None:
    #     for key, Twc in camera_poses_in_world_frame.items():
    #         body_name = f"{key}"
    #         body = etree.SubElement(worldbody, "body", name=body_name)
    #         body.set("pos", " ".join(map(str, Twc[:3, 3])))
    #         mj_correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    #         quat = Rotation.from_matrix(Twc[:3, :3] @ mj_correction).as_quat(scalar_first=True)
    #         body.set("quat", " ".join(map(str, quat)))

    #         etree.SubElement(body, "camera", name=key, fovy="45", mode="fixed")
    # # Alternatively, initialize default cameras
    # else:
    body_name = "global_0"
    body = etree.SubElement(worldbody, "body", name=body_name)
    body.set("pos", "0 0 1") #Position in WC of camera. We want to set this up somewhere
    body.set(
        "quat",
        "0.93521172660947172 0.35369745673769204 0.0097647345133944699 0.013482784182144731",
    ) # set it looking down 
    etree.SubElement(body, "camera", name="global_0", fovy="45", mode="fixed")
        # Can comment out the below code because it don't matter
        
    # Convert back to MuJoCo model
    xml_string = etree.tostring(root, encoding="unicode")
    new_model = mujoco.MjModel.from_xml_string(xml_string, assets)
    new_data = mujoco.MjData(new_model)

    return new_model, new_data


# def create_wrist_cameras_xml(model, assets):
#     """
#     Creates wrist cameras and returns updated MuJoCo model and data.

#     Args:
#         model: MuJoCo model to modify
#         assets: Model assets for XML parsing

#     Returns:
#         tuple: (Updated MjModel, New MjData)
#     """
#     # Save and parse XML
#     xml_path = "tmp.xml"
#     mujoco.mj_saveLastXML(xml_path, model)
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     # Add cameras
#     add_cameras_to_xml(root)

#     # Convert back to MuJoCo model
#     xml_string = ET.tostring(root, encoding="unicode")
#     new_model = mujoco.MjModel.from_xml_string(xml_string, assets)
#     new_data = mujoco.MjData(new_model)

#     return new_model, new_data


# def add_cameras_to_xml(root):
#     """
#     Adds wrist cameras to XML root element.

#     Args:
#         root: XML root element
#     """
#     worldbody = root.find("worldbody")
#     cam_params = get_camera_params()

#     # Add cameras to both hands
#     hands = {
#         "l_robot": find_hand(worldbody, "l_robot/hand"),
#         "r_robot": find_hand(worldbody, "r_robot/hand"),
#     }

#     for robot_prefix, hand in hands.items():
#         add_wrist_camera(hand, f"{robot_prefix}/wrist", cam_params)


def get_camera_params() -> Dict[str, str]:
    """
    Returns standard wrist camera parameters.

    Returns:
        dict: Camera parameters including position, orientation, mode, and FOV
    """
    return {
        "pos": "0 0 1",
        "quat": calculate_camera_quaternion(),
        "mode": "fixed",
        "fovy": "75",
    }


def calculate_camera_quaternion():
    """
    Calculates standard camera orientation quaternion.

    Returns:
        str: Space-separated quaternion values as string
    """
    final_rot = Rotation.from_euler("ZX", [90, 180 - 20], degrees=True)
    quat_wxyz = final_rot.as_quat(scalar_first=True).tolist()
    return " ".join(map(str, quat_wxyz))


def find_hand(worldbody, hand_name):
    """
    Finds hand body element in worldbody.

    Args:
        worldbody: XML worldbody element
        hand_name: Name of hand to find

    Returns:
        Element: Hand body element

    Raises:
        IndexError: If hand not found
    """
    hands = [body for body in worldbody.findall(".//body") if body.get("name") == hand_name]
    if not hands:
        raise IndexError(f"Hand '{hand_name}' not found in XML")
    return hands[0]


def add_wrist_camera(hand: ET.Element, camera_name: str, cam_params: dict) -> ET.Element:
    """
    Adds camera element to hand with given parameters.

    Args:
        hand: XML hand element to attach camera to
        camera_name: Name for new camera
        cam_params: Dictionary of camera parameters
    """
    # body = ET.SubElement(worldbody, "body", name="global_0")
    # # pos="-0.094 -0.292 0.698" quat="0.93521172660947172 0.35369745673769204 0.0097647345133944699 0.013482784182144731"
    # body.set("pos", "-0.094 -0.292 0.698")
    # body.set(
    #     "quat",
    #     "0.93521172660947172 0.35369745673769204 0.0097647345133944699 0.013482784182144731",
    # )

    # # Add camera to the body
    # camera = ET.SubElement(body, "camera", name="global_0")

    # First create a body for the camera
    body = ET.SubElement(
        hand, "body", name=camera_name, pos=cam_params["pos"], quat=cam_params["quat"]
    )

    camera = ET.SubElement(
        body, "camera", name=camera_name, mode=cam_params["mode"], fovy=cam_params["fovy"]
    )

    return camera


def transform_camera_calibration(
    model, data, camera_calibration: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Transforms camera calibration from robot frame to world frame.

    Args:
        camera_calibration: Camera calibration data

    Raises:
        ValueError: If camera calibration data is invalid
    """
    # Determine which robot frame the camera pose was calibrated in
    robot_name = (
        "l_robot/"
        if camera_calibration["extrinsics"]["in_robot_frame"].startswith("l")
        else "r_robot/"
    )
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_name)
    robot_base_body = data.body(id)
    robot_pose_in_world_frame = np.eye(4)
    robot_pose_in_world_frame[:3, :3] = Rotation.from_quat(
        robot_base_body.xquat, scalar_first=True
    ).as_matrix()
    robot_pose_in_world_frame[:3, 3] = robot_base_body.xpos
    # Transform from robot frame to world frame
    global_camera_poses_in_world_frame = {}
    for global_camera_key, v in camera_calibration["extrinsics"].items():
        if "global" in global_camera_key:
            camera_in_robot_frame = np.array(v)
            global_camera_poses_in_world_frame[global_camera_key] = (
                robot_pose_in_world_frame @ camera_in_robot_frame
            )
    return global_camera_poses_in_world_frame


import numpy as np
import cv2

def world_to_camera(point_world, camera_position, camera_rotation):
    """
    Transform a point from world coordinates to camera coordinates
    
    Parameters:
    - point_world: 3D point in world coordinates (np.array shape (3,))
    - camera_position: Camera position in world coordinates (np.array shape (3,))
    - camera_rotation: Camera rotation matrix (np.array shape (3, 3))
    
    Returns:
    - point_camera: 3D point in camera coordinates (np.array shape (3,))
    """
    # Translate point by camera position
    point_translated = point_world - camera_position
    
    # Rotate point by camera orientation
    point_camera = camera_rotation.T @ point_translated
    
    return point_camera

def camera_to_image(point_camera, camera_matrix):
    """
    Project a 3D point in camera coordinates to 2D image coordinates
    
    Parameters:
    - point_camera: 3D point in camera coordinates (np.array shape (3,))
    - camera_matrix: Camera intrinsic matrix (np.array shape (3, 3))
    
    Returns:
    - point_image: 2D point in image coordinates (np.array shape (2,))
    """
    # Check if point is in front of camera
    if point_camera[2] <= 0:
        return None
    
    # Project point to image plane
    point_normalized = point_camera / point_camera[2]
    
    # Apply camera intrinsic matrix
    point_image_homogeneous = camera_matrix @ point_normalized
    
    # Convert to 2D image coordinates
    point_image = point_image_homogeneous[:2]
    
    return point_image


def draw_ball_center(image, ball_center_world, camera_position, camera_rotation, camera_matrix):
    """
    Draw a point at the center of a ball in world space onto an image
    
    Parameters:
    - image: Input image
    - ball_center_world: Ball center in world coordinates (np.array shape (3,))
    - camera_position: Camera position in world coordinates (np.array shape (3,))
    - camera_rotation: Camera rotation matrix (np.array shape (3, 3))
    - camera_matrix: Camera intrinsic matrix (np.array shape (3, 3))
    
    Returns:
    - image: Image with ball center marked
    """
    # Convert ball center from world to camera coordinates
    ball_center_camera = world_to_camera(ball_center_world, camera_position, camera_rotation)
    
    # Project ball center to image plane
    ball_center_image = camera_to_image(ball_center_camera, camera_matrix)
    
    # Check if ball center is visible
    if ball_center_image is not None:
        # Convert to integer coordinates for drawing
        x, y = int(ball_center_image[0]), int(ball_center_image[1])
        
        # Check if point is within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # Draw circle at ball center
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red circle with radius 5
            
            # Draw crosshair
            cv2.line(image, (x - 10, y), (x + 10, y), (0, 0, 255), 2)
            cv2.line(image, (x, y - 10), (x, y + 10), (0, 0, 255), 2)
    
    return image

# Example usage
def main():
    # Create a test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Define camera parameters
    fx, fy = 379.004150390625, 378.7071228027344  # Focal lengths
    cx, cy = 313.1700439453125, 353.9  # Principal point
    
    # Camera intrinsic matrix
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Camera extrinsic parameters (position and orientation)
    camera_position = np.array([0, 0, 0])  # Camera at origin
    
    # Create rotation matrix (camera looking down the z-axis)
    camera_rotation = np.eye(3)  # Identity matrix for no rotation
    
    # Ball center in world coordinates (in front of camera)
    ball_center_world = np.array([0.5, 0.3, 3.0])  # 3 meters in front of camera
    
    # Draw ball center on image
    result = draw_ball_center(image, ball_center_world, camera_position, camera_rotation, camera_matrix)
    
    # Display result
    cv2.imshow('Image with Ball Center', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
