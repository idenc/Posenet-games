import math

import cv2
import numpy as np

import posenet.constants


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    target_width = target_height
    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    img = cap.read()
    left = (img.shape[1] // 2) - (img.shape[0] // 2)
    right = (img.shape[1] // 2) + (img.shape[0] // 2)
    img = img[:, left:right]
    if img is None:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def angle_between(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p1[1] - p2[1]
    theta_rad = math.atan2(delta_y, delta_x)
    return np.rad2deg(theta_rad)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def overlay_image_alpha(img, img_overlay, pos, angle):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """
    img_overlay = rotate_image(img_overlay, angle)

    alpha_mask = img_overlay[:, :, 3] / 255.0
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_coords(coords):
    return int(round(coords[1])), int(round(coords[0]))


def map_image(keypoint_coords, out_img):
    # Handle torso
    r_shoulder = keypoint_coords[6]
    l_shoulder = keypoint_coords[5]
    torso_coords = get_coords(r_shoulder)
    dist = np.linalg.norm(r_shoulder - l_shoulder)

    if dist > 0 and torso_coords != (0, 0):
        try:
            torso_img = image_resize(posenet.TORSO_IMG, width=int(round(dist)))
            delta_y = int(round(1 / 4 * torso_img.shape[1]))
        except:
            return out_img
        overlay_image_alpha(out_img,
                            torso_img,
                            (torso_coords[0], torso_coords[1] - delta_y),
                            -1 * angle_between(torso_coords,
                                               get_coords(l_shoulder)))
    # Handle face
    face_coord = get_coords(keypoint_coords[2])
    l_ear_coord = keypoint_coords[3]
    r_ear_coord = keypoint_coords[4]
    dist = np.linalg.norm(l_ear_coord - r_ear_coord)
    if dist > 0 and face_coord != (0, 0):
        try:
            face_img = image_resize(posenet.FACE_IMG, width=int(round(dist) + 15))
            delta_x = int(round(1 / 3 * face_img.shape[0]))
            delta_y = int(round(1 / 3 * face_img.shape[1]))
        except:
            return out_img
        overlay_image_alpha(out_img,
                            face_img,
                            (face_coord[0] - delta_x, face_coord[1] - delta_y),
                            -1 * angle_between(
                                get_coords(r_ear_coord),
                                get_coords(l_ear_coord)))

    # Handle right bicep
    r_elbow = keypoint_coords[8]
    r_bicep = get_coords(r_elbow)
    dist = np.linalg.norm(r_shoulder - r_elbow)
    if dist > 0 and r_bicep != (0, 0):
        try:
            r_bicep_img = image_resize(posenet.RIGHT_BICEP_IMG, height=int(round(dist)) + 15)
            delta_x = int(1 / 5 * round(r_bicep_img.shape[0]))
            delta_y = int(1.5 * round(r_bicep_img.shape[1]))
        except:
            return out_img
        overlay_image_alpha(out_img,
                            r_bicep_img,
                            (r_bicep[0] - 0, r_bicep[1] - delta_y),
                            -1 * angle_between(r_elbow,
                                               torso_coords))

    # Handle left bicep
    l_elbow = keypoint_coords[7]
    l_bicep = get_coords(keypoint_coords[5])
    dist = np.linalg.norm(keypoint_coords[5] - l_elbow)
    if dist > 0 and l_bicep != (0, 0):
        try:
            l_bicep_img = image_resize(posenet.LEFT_BICEP_IMG, height=int(round(dist)))
            delta_x = int(1 / 4 * round(l_bicep_img.shape[0]))
            delta_y = int(1 / 4 * round(l_bicep_img.shape[1]))
        except:
            return out_img
        overlay_image_alpha(out_img,
                            l_bicep_img,
                            (l_bicep[0] - delta_x, l_bicep[1] - delta_y),
                            angle_between(l_elbow, get_coords(keypoint_coords[7])))

    # Handle right hand
    r_wrist = keypoint_coords[10]
    dist = np.linalg.norm(r_elbow - r_wrist)
    r_hand_coords = get_coords(r_elbow)
    if dist > 0 and r_hand_coords != (0, 0):
        try:
            r_hand_img = image_resize(posenet.RIGHT_HAND_IMG, height=int(round(dist)))
            delta_x = int(1 / 3 * round(r_hand_img.shape[0]))
        except:
            return out_img
        overlay_image_alpha(out_img,
                            r_hand_img,
                            (r_hand_coords[0] - delta_x, r_hand_coords[1]),
                            angle_between(get_coords(keypoint_coords[8]), r_wrist))

    # Handle left hand
    l_wrist = keypoint_coords[9]
    dist = np.linalg.norm(keypoint_coords[7] - l_wrist)
    l_hand_coords = get_coords(l_elbow)
    if dist > 0 and l_hand_coords != (0, 0):
        try:
            l_hand_img = image_resize(posenet.LEFT_HAND_IMG, height=int(round(dist)) + 20)
            delta_x = int(1 / 3 * round(l_hand_img.shape[0]))
        except:
            return out_img
        angle = angle_between(l_wrist, get_coords(keypoint_coords[7])) - 60
        overlay_image_alpha(out_img,
                            l_hand_img,
                            (l_hand_coords[0] - delta_x, l_hand_coords[1]),
                            angle)
    return out_img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    # out_img = map_image(keypoint_coords[0], out_img)
    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(0, 255, 255), thickness=2)
    return out_img
