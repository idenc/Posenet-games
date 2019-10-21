import cv2
import deepviewrt as rt
from deepviewrt.context import Context

import posenet as p
# import posenet.posenet_c.posenet as p_c
from WebcamCapture import WebcamVideoStream


def image_to_ratio(h, w):
    if (w / h) < (16 / 9):
        w, h = 16 * (w // 16), 9 * (w // 16)
    else:
        w, h = 16 * (h // 9), 9 * (h // 9)
    return w, h


class posenetInterface:
    def __init__(self, image_size):
        self.output_stride = 16

        print('DeepViewRT %s' % rt.version())

        self.cap = cv2.VideoCapture(0)
        in_width = self.cap.get(3)
        in_height = self.cap.get(4)
        print(in_width, in_height)
        self.cam_width, self.cam_height = image_to_ratio(in_height, in_width)
        # self.scale_factor = (270 / self.cam_height)

        self.cap.set(3, 480)
        self.cap.set(4, 480)
        self.scale_factor = (257 / 480)
        self.video = WebcamVideoStream(self.cap).start()

        rtm_file = "/home/pi/Downloads/posenet_050_257.rtm"
        self.client = Context()
        in_file = open(rtm_file, 'rb')
        self.client.load(in_file.read())

    def infer_image(self, num_people, return_overlay=False, return_input_img=False):
        input_image, display_image, output_scale = p.read_cap(
            self.video, scale_factor=self.scale_factor, output_stride=self.output_stride)

        inputs = {'input': input_image}
        self.client.run(inputs)

        heatmaps_result = self.client.tensor('output1').map()
        offsets_result = self.client.tensor('output2').map()
        displacement_fwd_result = self.client.tensor('output3').map()
        displacement_bwd_result = self.client.tensor('output4').map()

        pose_scores, keypoint_scores, keypoint_coords = p.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=self.output_stride,
            max_pose_detections=num_people,
            min_pose_score=0.20)
        keypoint_coords *= output_scale

        # for pose in keypoint_coords:
        #   pose[:, 1] += (display_image.shape[1] // 2) - (display_image.shape[0] // 2)

        if not return_overlay:
            return keypoint_coords[:num_people]

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = p.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.20, min_part_score=0.1)

        if return_input_img:
            return cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB), keypoint_coords[:num_people], input_image
        return cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB), keypoint_coords[:num_people]

    def get_image(self, num_people=1, return_input_img=False):
        return self.infer_image(num_people, return_overlay=True, return_input_img=return_input_img)

    def get_keypoints(self, num_people=1):
        return self.infer_image(num_people, return_overlay=False)
