"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution,"camera_depths":True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img

def get_libero_depth_image(obs):

    if "agentview_depth" not in obs:
        raise KeyError("Missing 'agentview_depth' in observations")

    depth_img = obs["agentview_depth"]
    depth_img = depth_img[::-1, ::-1]  # rotate 180 degrees to match RGB preprocessing
    return depth_img

def get_libero_wrist_depth_image(obs):

    if "robot0_eye_in_hand_depth" not in obs:
        raise KeyError("Missing 'robot0_eye_in_hand_depth' in observations")

    depth_img = obs["robot0_eye_in_hand_depth"]
    depth_img = depth_img[::-1, ::-1]  # rotate 180 degrees to match RGB preprocessing
    return depth_img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    #rollout_dir = f"./rollouts/libero-80"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    #mp4_path = f"{rollout_dir}/episode={idx}--success={success}--{task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


instruction_cot_dict = {
    # libero-goal
    "open the middle drawer of the cabinet": ['locate middle drawer', 'grasp drawer handle', 'pull open drawer'],
    "open the top drawer and put the bowl inside": ['locate top drawer', 'open top drawer by pulling handle', 'grab bowl', 'place bowl inside opened drawer'],
    "push the plate to the front of the stove": ['locate plate on stove', 'push plate toward front of stove'],
    'put the bowl on the plate': ['locate bowl', 'grab bowl by rim', 'place bowl at center of plate'],
    'put the bowl on the stove': ['locate bowl', 'grab bowl by rim', 'place bowl at a stable place on stove'],
    'put the bowl on top of the cabinet': ['locate bowl', 'grab bowl by rim', 'place bowl on cabinet top in the middle'],
    'put the cream cheese in the bowl': ['locate cream cheese', 'grab cream cheese', 'place into center of bowl'],
    'put the wine bottle on the rack': ['locate wine bottle', 'grab wine bottle by center', 'place carefully on rack'],
    'put the wine bottle on top of the cabinet': ['locate wine bottle', 'grab wine bottle by center', 'place carefully on cabinet top'],
    'turn on the stove': ['locate stove knob', 'rotate knob to turn on stove'],

    # libero-10
    "put both the alphabet soup and the tomato sauce in the basket": ['grab alphabet soup', 'place into basket', 'grab the tomato sauce', 'place into basket'],
    "put both the cream cheese box and the butter in the basket": ['grab cream cheese box', 'avoid obstacles, place into basket', 'grab butter', 'place into basket'],
    "turn on the stove and put the moka pot on it": ['find stove', 'rotate stove switch', 'grab moka pot', 'place above stove'],
    'put the black bowl in the bottom drawer of the cabinet and close it': ['grab black bowl', 'place in bottom drawer', 'close cabinet door'],
    'put the white mug on the left plate and put the yellow and white mug on the right plate': ['grab white mug', 'place on left plate', 'grab yellow and white mug', 'place on right plate'],
    'pick up the book and place it in the back compartment of the caddy': ['grab book', 'move to back of caddy', 'place into back compartment'],
    'put the white mug on the plate and put the chocolate pudding to the right of the plate': ['grab white mug', 'place on plate', 'grab chocolate pudding', 'place to right of plate'],
    'put both the alphabet soup and the cream cheese box in the basket': ['grab alphabet soup', 'place into basket', 'grab cream cheese box', 'place into basket'],
    'put both moka pots on the stove': ['grab moka pot close to stove', 'place on the stove. leave room for second moka pot.', 'grab second moka pot', 'place on stove'],
    'put the yellow and white mug in the microwave and close it': ['grab yellow mug', 'place inside microwave', 'close microwave door'],

    # libero-spatial
    "pick up the black bowl between the plate and the ramekin and place it on the plate": ['locate black bowl between plate and ramekin', 'grab black bowl by rim', 'place it in middle of plate'],
    "pick up the black bowl from table center and place it on the plate": ['locate black bowl at center of table', 'grab black bowl by rim', 'place it at center of plate'],
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": ['locate black bowl in top drawer', 'grab black bowl while avoiding obstacles', 'place bowl in middle of plate'],
    'pick up the black bowl next to the cookie box and place it on the plate': ['locate the black bowl next to cookie box', 'grab the black bowl by the rim', 'place it at the center of the plate'],
    'pick up the black bowl next to the plate and place it on the plate': ['find black bowl next to plate', 'grab black bowl by rim', 'place bowl in middle of plate'],
    'pick up the black bowl next to the ramekin and place it on the plate': ['grab black bowl next to ramekin', 'place it at center of plate'],
    'pick up the black bowl on the cookie box and place it on the plate': ['find black bowl on top of cookie box', 'grab black bowl', 'place bowl in middle of plate'],
    'pick up the black bowl on the ramekin and place it on the plate': ['locate black bowl on ramekin', 'grab black bowl', 'place it at center of plate'],
    'pick up the black bowl on the stove and place it on the plate': ['find black bowl on stove', 'grab black bowl', 'place bowl in middle of plate'],
    'pick up the black bowl on the wooden cabinet and place it on the plate': ['find black bowl placed on top of wooden drawer', 'grab black bowl', 'place it at center of plate'],

    # libero-object
    "pick up the alphabet soup and place it in the basket": ['locate alphabet soup', 'grab alphabet soup with a stable grasp', 'place into center of basket'],
    "pick up the bbq sauce and place it in the basket": ['locate bbq sauce bottle', 'grab bbq sauce', 'drop into basket'],
    "pick up the butter and place it in the basket": ['locate butter', 'grab butter', 'place into basket'],
    'pick up the chocolate pudding and place it in the basket': ['locate chocolate pudding', 'grab chocolate pudding', 'drop into basket'],
    'pick up the cream cheese and place it in the basket': ['locate cream cheese', 'grab cream cheese', 'place into basket'],
    'pick up the ketchup and place it in the basket': ['locate red ketchup bottle', 'grab ketchup', 'place into center of basket'],
    'pick up the milk and place it in the basket': ['locate milk container', 'grab milk', 'place into basket'],
    'pick up the orange juice and place it in the basket': ['locate orange juice container', 'grab orange juice', 'place into center of basket'],
    'pick up the salad dressing and place it in the basket': ['locate salad dressing on table', 'grab salad dressing', 'place into basket'],
    'pick up the tomato sauce and place it in the basket': ['locate red tomato sauce container', 'grab tomato sauce', 'place into center of basket']
}
