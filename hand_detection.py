import cv2
import numpy as np
import pygame
import time
from tensorflow.lite.python.interpreter import Interpreter

import posenet_interface

CONNECTED_POINTS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 17), (0, 5), (13, 17), (9, 13), (5, 9), (17, 18), (18, 19),
                    (19, 20), (13, 14), (14, 15), (15, 16), (9, 10), (10, 11), (11, 12), (5, 6), (6, 7), (7, 8)]


class Gesture:
    def __init__(self, width=None, height=None, pnet=None):
        pygame.init()
        self.clock = pygame.time.Clock()

        if width and height:
            self.WIDTH = width
            self.HEIGHT = height
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            info_object = pygame.display.Info()
            self.WIDTH = info_object.current_w
            self.HEIGHT = info_object.current_h
            flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
            self.window = pygame.display.set_mode(flags=flags)
        pygame.display.set_caption('Posenet Hands')

        if pnet:
            self.pnet = pnet
        else:
            self.pnet = posenet_interface.posenetInterface(257)

        self.myfont = pygame.font.SysFont("Comic Sans MS", 40)
        self.image = None
        self.interpreter = Interpreter(model_path='hand_landmark.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self):
        while 1:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            pygame.display.update()
            self.clock.tick(60)

    def blit_cam_frame(self, frame, screen):
        # frame = np.fliplr(frame)
        frame = np.rot90(frame)

        frame = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(frame, (self.WIDTH, self.HEIGHT)), (0, 0))

    def scale_keypoints(self, original_shape, keypoints):
        scale = np.array([self.HEIGHT / original_shape[0], self.WIDTH / original_shape[1]])
        return keypoints * scale

    def get_adjacent_keypoints(self, keypoints):
        results = []
        for left, right in CONNECTED_POINTS:
            results.append(
                np.array([keypoints[left].pt, keypoints[right].pt]).astype(np.int32),
            )
        return results

    def hand_landmark(self, input_image, right_hand):
        target_width = target_height = 256
        input_image = input_image[0]
        scale_x = self.image.shape[1] / 256
        scale_y = self.image.shape[0] / 256
        shift_x = 0
        shift_y = 0
        # If hand keypoint is available
        if right_hand[0] != 0.0 or right_hand[1] != 0.0:
            right_hand = np.int0(right_hand)
            # Take square region near hand keypoint which is located at wrist
            square_size = 150
            square_size_half = square_size // 2
            shift_x = max(right_hand[1] - square_size_half, 0)
            shift_y = max(right_hand[0] - square_size, 0)
            input_image = self.image[shift_y:right_hand[0],
                            shift_x:right_hand[1] + square_size_half]

            scale_x = input_image.shape[1] / 256
            scale_y = input_image.shape[0] / 256

            # TODO Find hand and rotate so hand is always upright
            # input_image = find_hand(new_img)
            if input_image is None:
                return
            # cv2.imshow('hand', input_image)
            input_image = input_image * (2.0 / 255.0) - 1.0

        input_img = cv2.resize(input_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR).astype(
            np.float32)
        input_img = np.expand_dims(input_img, 0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
        start = time.time()
        self.interpreter.invoke()
        print('infer time:', time.time() - start)
        hand_points = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        hand_confidence = self.interpreter.get_tensor(self.output_details[1]['index'])
        if hand_confidence[0] < 0.1:
            return
        norm_hand_points = []
        for i in range(0, len(hand_points), 2):
            x = (hand_points[i] * scale_x) + shift_x
            y = (hand_points[i + 1] * scale_y) + shift_y
            # cv2.putText(self.image, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 0, 0))
            norm_hand_points.append(cv2.KeyPoint(x, y, 10))
        return norm_hand_points

    def draw(self):
        self.image, keypoints, input_image = self.pnet.get_image(return_input_img=True)
        norm_hand_points = self.hand_landmark(input_image, keypoints[0][10])
        if norm_hand_points is not None:
            self.image = cv2.drawKeypoints(
                self.image, norm_hand_points, outImage=np.array([]), color=(0, 255, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            adjacent_keypoints = self.get_adjacent_keypoints(norm_hand_points)
            self.image = cv2.polylines(self.image, adjacent_keypoints, isClosed=False, color=(0, 255, 255), thickness=2)

        self.blit_cam_frame(self.image, self.window)
        frames = self.myfont.render(str(int(self.clock.get_fps())) + " FPS", True, pygame.Color('green'))
        self.window.blit(frames, (self.WIDTH - 100, 20))


if __name__ == "__main__":
    g = Gesture(480, 480)
    g.run()
