import numpy as np
from enum import IntEnum


class Keypoints(IntEnum):
    """
    This class defines the indices for each body part's keypoint
    Usage: left_hand = keypoint[Keypoints.LEFT_HAND]
    and left_hand is a list with 2 element [LEFT_HAND_Y, LEFT_HAND_X]

    ~Note~
    (0, 0) is at the top left corner of the window and x increases to the right, but
    y increases as you move down instead of the usual increase as y goes up.
    """
    NOSE = 0

    LEFT_EYE = 1
    RIGHT_EYE = 2

    LEFT_EAR = 3
    RIGHT_EAR = 4

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8

    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    LEFT_HIP = 11
    RIGHT_HIP = 12

    LEFT_KNEE = 13
    RIGHT_KNEE = 14

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


def euclidean_distance(p1, p2):
    """
    Helper function to find the distance between two points
    """
    return np.linalg.norm(p1 - p2)


def check_jump(frames):
    """
    frames contains the keypoints that were detected
    by the posenet model for each frame since the last gesture checks.
    Thus, you should loop through the frames and check for any gestures.
    :param frames: List of keypoints detected for each frame since
                   last time this function was called
    :return: True if gesture is detected, else false
    """
    for keypoints in frames:
        return


def check_run(frames):
    """
    frames contains the keypoints that were detected
    by the posenet model for each frame since the last gesture checks.
    Thus, you should loop through the frames and check for any gestures.
    :param frames: List of keypoints detected for each frame since
                   last time this function was called
    :return: True if gesture is detected, else false
    """
    for keypoints in frames:
        return
