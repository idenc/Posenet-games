import random
import pygame
import numpy as np
import sys
from threading import Thread, enumerate

import posenet_interface
from pygame import *

WHITE = (255, 255, 255)
ORANGE = (255, 140, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def get_coords(coords):
    return [int(round(coords[1])), int(round(coords[0]))]


class Dodger:

    def __init__(self, num_boxes, width=None, height=None, pnet=None):
        pygame.init()
        self.clock = pygame.time.Clock()

        self.image = None

        self.NUM_BOXES = num_boxes
        self.score = -1
        self.failed = False
        self.fail_time = 0
        self.warn = True
        self.warn_time = 0
        self.warn_img = pygame.image.load('exclamation_mark.png')
        self.rock_img = pygame.image.load('rock.png')
        self.gravity = 20
        self.box_speed = 0
        self.delta_time = self.clock.get_time()

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
        pygame.display.set_caption('Posenet Dodger')

        self.boxes = None
        self.frame_count = 0
        self.time_frame = 0

        if pnet:
            self.pnet = pnet
        else:
            self.pnet = posenet_interface.posenetInterface(257)
            
        self.image, _ = self.pnet.get_image()

    def blit_cam_frame(self, frame, screen):
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(frame, (self.WIDTH, self.HEIGHT)), (0, 0))

    def run(self):
        # Main game loop
        while True:
            # Draw objects
            self.draw()

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        return

            # Update screen
            pygame.display.update()
            self.clock.tick(60)  # Limit to 60 FPS

    def scale_keypoints(self, original_shape, keypoints):
        scale = np.array([self.HEIGHT / original_shape[0], self.WIDTH / original_shape[1]])
        return keypoints * scale

    def infer(self, canvas):
        self.image, keypoints = self.pnet.get_image()
        keypoints = self.scale_keypoints(self.image.shape, keypoints[0])

        for point in keypoints:
            pos = get_coords(point)
            pos = (self.WIDTH - pos[0], pos[1])
            bounding_box = pygame.Rect(pos, (10, 10))
            bounding_box.center = pos
            canvas.fill(WHITE, bounding_box)

    def check_collisions(self):
        self.image, keypoints = self.pnet.get_image()
        print("Img shape: ", self.image.shape)
        keypoints = self.scale_keypoints(self.image.shape, keypoints[0])

        for point in keypoints:
            pos = get_coords(point)
            pos = (self.WIDTH - pos[0], pos[1])
            bounding_box = pygame.Rect(pos, (10, 10))
            self.window.fill(WHITE, bounding_box)
            bounding_box.center = pos
            if bounding_box.collidelist(self.boxes) != -1:
                self.score = 0
                self.failed = True
                self.fail_time = pygame.time.get_ticks()
                return

    def make_warning(self):
        self.warn = True
        self.warn_time = pygame.time.get_ticks()

    def gen_boxes(self):
        self.boxes = []
        self.box_speed = 20 * self.score
        if not self.failed:
          self.score += 1
        for i in range(self.NUM_BOXES):
            x = random.randrange(0, self.WIDTH - 200)
            delta_x = random.randrange(20, self.HEIGHT // 4)
            delta_y = random.randrange(20, self.WIDTH // 4)
            rect = pygame.Rect(x, -delta_y, delta_x, delta_y)
            rect.bottom = -delta_y
            self.boxes.append(rect)

    def update_boxes(self, canvas):
        self.box_speed += self.gravity
        self.delta_time = self.clock.get_time() / 1000
        new_boxes = []
        for box in self.boxes:
            box.y += int(self.box_speed * self.delta_time)
            if box.midtop[1] < self.HEIGHT:
                new_boxes.append(box)
                canvas.blit(pygame.transform.scale(self.rock_img, (box.width, box.height)), box)
        return new_boxes

    def draw(self):
        fps = self.clock.get_fps()

        if not self.boxes:
            self.make_warning()
            self.gen_boxes()

        # canvas.fill(BLACK)
        if self.frame_count % 2 == 0 and len(enumerate()) == 2:
            thread = Thread(target=self.check_collisions)
            thread.daemon = True
            thread.start()

        self.blit_cam_frame(self.image, self.window)

        if self.failed:
            myfont1 = pygame.font.SysFont("Comic Sans MS", 180)
            fail_label = myfont1.render("FAILED", 1, (255, 0, 0))
            fail_rect = fail_label.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.window.blit(fail_label, fail_rect)

            if pygame.time.get_ticks() - self.fail_time >= 1000:
                self.failed = False

        if self.warn:
            for box in self.boxes:
                self.window.blit(self.warn_img, [box.left, 0])

            if pygame.time.get_ticks() - self.warn_time >= (4000 - 400 * self.score):
                self.warn = False
        else:
            self.boxes = self.update_boxes(self.window)

        myfont1 = pygame.font.SysFont("Comic Sans MS", self.HEIGHT // 20)
        fail_label = myfont1.render("Score " + str(self.score), 1, (255, 255, 0))
        self.window.blit(fail_label, (50, 20))

        frames = myfont1.render(str(int(fps)) + " FPS", True, pygame.Color('white'))
        self.window.blit(frames, (50, 50))

        self.frame_count += 1


if __name__ == "__main__":
    dodge = Dodger(3)
    dodge.run()
