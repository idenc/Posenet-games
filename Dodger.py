import random
import pygame
import numpy as np
import sys

import posenet_interface
from pygame import *

WHITE = (255, 255, 255)
ORANGE = (255, 140, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def get_coords(coords):
    return int(round(coords[1])), int(round(coords[0]))


class Dodger:

    def __init__(self, num_boxes, width=None, height=None):
        pygame.init()
        self.clock = pygame.time.Clock()

        self.NUM_BOXES = num_boxes
        self.score = 0
        self.failed = False
        self.fail_time = 0

        if width and height:
            self.WIDTH = width
            self.HEIGHT = height
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            infoObject = pygame.display.Info()
            self.WIDTH = infoObject.current_w
            self.HEIGHT = infoObject.current_h
            flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
            self.window = pygame.display.set_mode(flags=flags)
        pygame.display.set_caption('Posenet Dodger')

        self.boxes = None
        self.gen_boxes()
        self.frame_count = 0
        self.time_frame = 0
        self.pnet = posenet_interface.posenetInterface(257)

    def blit_cam_frame(self, frame, screen):
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(frame, (self.WIDTH, self.HEIGHT)), (0, 0))

    def run(self):
        # Main game loop
        while True:
            # Draw objects
            self.draw(self.window)

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
        image, keypoints = self.pnet.get_image()
        keypoints = self.scale_keypoints(image.shape, keypoints)
        self.blit_cam_frame(image, canvas)

        for point in keypoints:
            pos = get_coords(point)
            pos = (self.WIDTH - pos[0], pos[1])
            bounding_box = pygame.Rect(pos, (10, 10))
            bounding_box.center = pos
            canvas.fill(WHITE, bounding_box)

    def check_collisions(self, canvas):
        image, keypoints = self.pnet.get_image()
        keypoints = self.scale_keypoints(image.shape, keypoints)
        self.blit_cam_frame(image, canvas)

        for point in keypoints:
            pos = get_coords(point)
            pos = (self.WIDTH - pos[0], pos[1])
            bounding_box = pygame.Rect(pos, (10, 10))
            canvas.fill(WHITE, bounding_box)
            bounding_box.center = pos
            if bounding_box.collidelist(self.boxes) != -1:
                self.score = 0
                self.failed = True
                self.fail_time = pygame.time.get_ticks()
                return
        self.score += 1

    def gen_boxes(self):
        self.boxes = []
        for i in range(self.NUM_BOXES):
            x = random.randrange(0, self.WIDTH - 200)
            y = random.randrange(0, self.HEIGHT - 200)
            delta_x = random.randrange(20, self.WIDTH // 10)
            delta_y = random.randrange(20, self.HEIGHT // 10)
            self.boxes.append(pygame.Rect(x, y, delta_x, delta_y))

    def draw(self, canvas):
        fps = self.clock.get_fps()

        # canvas.fill(BLACK)
        if self.frame_count % 2 == 0:  # Infer every second frame
            # Check collisions and generate new boxes based on the fps
            if self.frame_count == 200:
                self.check_collisions(canvas)
                self.gen_boxes()
                self.frame_count = 0
            else:
                self.infer(canvas)

            for box in self.boxes:
                # Grow boxes to make them seem like they are coming closer
                box.inflate_ip(2, 2)
                canvas.fill(GREEN, box)

        if self.failed:
            myfont1 = pygame.font.SysFont("Comic Sans MS", 50)
            label1 = myfont1.render("FAILED", 1, (255, 0, 0))
            canvas.blit(label1, (self.WIDTH // 2, self.HEIGHT // 2))

            if pygame.time.get_ticks() - self.fail_time >= 1000:
                self.failed = False

        myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
        label1 = myfont1.render("Score " + str(self.score), 1, (255, 255, 0))
        canvas.blit(label1, (50, 20))
        label1 = myfont1.render(str(200 - self.frame_count), 1, (255, 255, 0))
        canvas.blit(label1, (200, 20))

        frames = myfont1.render(str(int(fps)) + " FPS", True, pygame.Color('white'))
        canvas.blit(frames, (50, 50))

        self.frame_count += 1


if __name__ == "__main__":
    dodge = Dodger(num_boxes=3)
    dodge.run()
