import numpy as np
import posenet_interface
import pygame
import subprocess
import sys
import time as ti

from pygame import *
from pynput.keyboard import Key, Controller
from posenet.utils import angle_between


def sign(x):
    return 1 - (x <= 0)


class Gesture:
    def __init__(self, game, width=None, height=None, pnet=None):
        pygame.init()
        self.clock = pygame.time.Clock()
        if len(sys.argv > 1):
            self.buffer_time = int(sys.argv[1])
        else:
            self.buffer_time = 300

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
        pygame.display.set_caption('Posenet Gestures')

        if pnet:
            self.pnet = pnet
        else:
            self.pnet = posenet_interface.posenetInterface(257)

        self.keypoints = []
        time = pygame.time.get_ticks()
        self.check_time = time
        self.myfont = pygame.font.SysFont("Comic Sans MS", 40)
        self.buffer_time = time
        self.jump_time = time
        self.image = None
        self.gesture = []
        self.game = game
        self.keyboard = Controller()
        self.start = True

        subprocess.Popen(["/usr/games/mednafen", "-psx.dbg_level", "0", "-video.fs", "0", "-cheats", "1",
                          "/home/pi/Downloads/NES_Roms/" + game + ".nes"])
        ti.sleep(1)
        with self.keyboard.pressed(Key.ctrl):
            with self.keyboard.pressed(Key.alt):
                self.keyboard.press(Key.left)
                self.keyboard.release(Key.left)
        # self.keyboard.press('d') # Hold down speed

    def run(self):
        # with self.keyboard.pressed(Key.ctrl):
        #     with self.keyboard.pressed(Key.alt):
        #         self.keyboard.press(Key.right)
        #         self.keyboard.release(Key.right)
        while 1:
            if self.start:
                self.draw()
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    raise SystemExit
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        return
                    elif event.key == K_TAB:
                        self.start = True

            pygame.display.update()
            self.clock.tick(60)

    def blit_cam_frame(self, frame, screen):
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(frame, (self.WIDTH, self.HEIGHT)), (0, 0))

    def scale_keypoints(self, original_shape, keypoints):
        scale = np.array([self.HEIGHT / original_shape[0], self.WIDTH / original_shape[1]])
        return keypoints * scale

    def draw(self):
        self.image, keypoints = self.pnet.get_image()

        self.blit_cam_frame(self.image, self.window)
        frames = self.myfont.render(str(int(self.clock.get_fps())) + " FPS", True, pygame.Color('green'))
        self.window.blit(frames, (self.WIDTH - 100, 20))
        if len(self.gesture) > 0:
            self.show_gesture(self.gesture)

        self.keypoints.append(keypoints[0])

        if pygame.time.get_ticks() - self.check_time >= self.buffer_time:
            self.gesture = []
            self.check_time = pygame.time.get_ticks()
            self.check_crouch()
            self.check_jump()
            self.check_run()
            self.check_wave('left')
            self.check_wave('right')
            self.check_hips()
            del self.keypoints[:len(self.keypoints) - 1]
            if self.game == "Tetris":
                self.release_keys()
            elif pygame.time.get_ticks() - self.buffer_time >= self.buffer_time + 50:
                self.buffer_time = pygame.time.get_ticks()

                if "RUNNING LEFT" not in self.gesture:
                    self.keyboard.release('a')
                if "RUNNING RIGHT" not in self.gesture:
                    self.keyboard.release('d')
                if "RIGHT WAVE" not in self.gesture:
                    self.keyboard.release('w')
                if "LEFT WAVE" not in self.gesture:
                    self.keyboard.release('g')
                if "CROUCH" not in self.gesture:
                    self.keyboard.release('s')
                if "ENTER" not in self.gesture:
                    self.keyboard.release(Key.enter)

            if pygame.time.get_ticks() - self.jump_time >= self.buffer_time + 200:
                self.jump_time = pygame.time.get_ticks()
                if "JUMP" not in self.gesture:
                    self.keyboard.release('f')

    def release_keys(self):
        if "RUNNING LEFT" in self.gesture:
            self.keyboard.release('a')
        if "RUNNING RIGHT" in self.gesture:
            self.keyboard.release('d')
        if "RIGHT WAVE" in self.gesture:
            self.keyboard.release('w')
        if "LEFT WAVE" in self.gesture:
            self.keyboard.release('g')
        if "JUMP" in self.gesture:
            self.keyboard.release('f')
        if "CROUCH" in self.gesture:
            self.keyboard.release('s')
        if "ENTER" in self.gesture:
            self.keyboard.release(Key.enter)

    def check_run(self):
        if not any("RUNNING" in string for string in self.gesture):
            for frame in self.keypoints:
                l_shoulder = frame[5]
                r_shoulder = frame[6]
                l_elbow = frame[7]
                r_elbow = frame[8]
                l_wrist = frame[9]
                r_wrist = frame[10]
                delta = np.linalg.norm(l_shoulder - r_shoulder)
                dir = ""
                # Check that wrist x coord is at least delta distance from shoulder x coord
                if abs(l_shoulder[1] - l_wrist[1]) > delta:  # Check left arm
                    dir += "LEFT"
                if abs(r_shoulder[1] - r_wrist[1]) > delta:  # Check right arm
                    dir += "RIGHT"
                if dir != "" and dir != "LEFTRIGHT":
                    if dir == "LEFT":
                        if l_wrist[0] < l_elbow[0]:
                            self.jump()
                        self.keyboard.press('a')
                    elif dir == "RIGHT":
                        if r_wrist[0] < r_elbow[0]:
                            self.jump()
                        self.keyboard.press('d')
                    run_str = "RUNNING " + dir
                    if run_str not in self.gesture:
                        self.gesture.append(run_str)
                    print(run_str)
                    return

    def check_wave(self, hand):
        if hand == 'left':
            hand_index = 9
            elbow_index = 7
        else:
            hand_index = 10
            elbow_index = 8

        start_wave = False
        for frame in self.keypoints:
            hand_coords = frame[hand_index]
            elbow_coords = frame[elbow_index]
            # Hand above elbow
            if hand_coords[0] < elbow_coords[0]:
                start_wave = True

        # Check that hand is above elbow
        if start_wave and len(self.keypoints) > 2:
            num_direction_changes = 0
            for i in range(len(self.keypoints) - 3):
                if sign(self.keypoints[i][hand_index][1] - self.keypoints[i + 1][hand_index][1]) != sign(
                        self.keypoints[i + 1][hand_index][1] - self.keypoints[i + 2][hand_index][1]):
                    num_direction_changes += 1
            if num_direction_changes > 0 and "JUMP" not in self.gesture:
                hand_str = hand.upper() + " WAVE"
                if hand_str not in self.gesture:
                    self.gesture.append(hand_str)
                print(hand_str)
                if hand == 'right':
                    self.keyboard.press('w')
                else:
                    self.keyboard.press('g')

    def check_crouch(self):
        for frame in self.keypoints:
            l_hand_y = frame[9][0]
            r_hand_y = frame[10][0]
            l_knee_y = frame[13][0]
            r_knee_y = frame[14][0]
            l_dist = abs(l_hand_y - l_knee_y)
            r_dist = abs(r_hand_y - r_knee_y)
            delta = self.HEIGHT // 7
            if l_hand_y != 0.0 and r_hand_y != 0.0 and l_dist <= delta and r_dist <= delta:
                if "CROUCH" not in self.gesture:
                    self.gesture.append("CROUCH")
                print("CROUCH")
                self.keyboard.press('s')
                return

    def jump(self):
        if "JUMP" not in self.gesture:
            self.gesture.append("JUMP")
        print("JUMP")
        self.keyboard.press('f')

    def check_jump(self):
        if not any("RUNNING" in string for string in self.gesture):
            for frame in self.keypoints:
                l_hand_y = frame[9][0]
                r_hand_y = frame[10][0]
                l_elbow_y = frame[7][0]
                r_elbow_y = frame[8][0]
                if l_hand_y != 0.0 and r_hand_y != 0.0 and l_hand_y < l_elbow_y and r_hand_y < r_elbow_y:
                    self.jump()
                    return

    def check_hips(self):
        hip_pose = True
        for frame in self.keypoints:
            l_wrist = frame[9]
            r_wrist = frame[10]
            l_hip = frame[11]
            r_hip = frame[12]
            l_elbow = frame[7]
            r_elbow = frame[8]
            l_dist = np.linalg.norm(l_hip - l_wrist)
            r_dist = np.linalg.norm(r_wrist - r_hip)
            if l_dist > 60 or r_dist > 60 or \
                    angle_between(l_wrist[::-1], l_elbow[::-1]) > 70 or \
                    angle_between(r_wrist[::-1], r_elbow[::-1]) < 100:
                hip_pose = False

        if hip_pose:
            if "ENTER" not in self.gesture:
                self.gesture.append("ENTER")
            print("ENTER")
            self.keyboard.press(Key.enter)

    def show_gesture(self, gesture):
        y_height = 20
        for g in gesture:
            gesture_label = self.myfont.render(g, 1, (0, 255, 0))
            self.window.blit(gesture_label, (50, y_height))
            y_height += self.myfont.get_height()


if __name__ == "__main__":
    g = Gesture('SMB', 480, 480)
    g.run()
