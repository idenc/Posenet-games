# Code adapted from https://github.com/hamdyaea/Daylight-Pong-python3
import argparse
import random
import pygame
import numpy as np
import sys

import posenet_interface
from pygame import *
from threading import Thread, enumerate

WHITE = (255, 255, 255)
ORANGE = (255, 140, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def get_coords(coords):
    return [int(round(coords[1])), int(round(coords[0]))]


def sign(x):
    return 1 - (x <= 0)


def keydown(event):
    global paddle1_vel, paddle2_vel

    if event.key == K_UP:
        paddle2_vel = -8
    elif event.key == K_DOWN:
        paddle2_vel = 8
    elif event.key == K_z:
        paddle1_vel = -8
    elif event.key == K_s:
        paddle1_vel = 8


def keyup(event):
    global paddle1_vel, paddle2_vel

    if event.key in (K_z, K_s):
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        paddle2_vel = 0


class Pong:
    def __init__(self, two_player, width=None, height=None, pnet=None):
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
        pygame.display.set_caption('Posenet Pong')

        self.BALL_RADIUS = self.HEIGHT // 24
        self.PAD_WIDTH = self.WIDTH // 35
        self.PAD_HEIGHT = self.HEIGHT // 4
        self.HALF_PAD_WIDTH = self.PAD_WIDTH // 2
        self.HALF_PAD_HEIGHT = self.PAD_HEIGHT // 2

        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.vel_time = pygame.time.get_ticks()

        self.frame_count = 2

        if pnet:
            self.pnet = pnet
        else:
            self.pnet = posenet_interface.posenetInterface(257)
            
        self.image, _ = self.pnet.get_image()

        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1, self.HEIGHT // 2]
        self.paddle2_pos = [self.WIDTH + 1 - self.HALF_PAD_WIDTH, self.HEIGHT // 2]
        self.l_score = 0
        self.r_score = 0
        self.two_player = two_player

        if random.randrange(0, 2) == 0:
            self.ball_init(True)
        else:
            self.ball_init(False)

    def run(self):
        while 1:
            self.draw()

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        return

            pygame.display.update()
            self.clock.tick(60)

    def ball_init(self, right):
        self.ball_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        # Random ball beginning speed and direction
        horz = random.randrange(120, 160)
        vert = random.randrange(-120, 120)

        if not right:
            horz = - horz

        self.ball_vel = [horz, -vert]

    def blit_cam_frame(self, frame, screen):
        # frame = np.fliplr(frame)
        frame = np.rot90(frame)
    
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(frame, (self.WIDTH, self.HEIGHT)), (0, 0))

    def scale_keypoints(self, original_shape, keypoints):
        scale = np.array([self.HEIGHT / original_shape[0], self.WIDTH / original_shape[1]])
        return keypoints * scale

    def move_paddle_sp(self, keypoints):
        try:
            # Get body coords if they exist
            bottom = get_coords(keypoints[6])[1]
            top = get_coords(keypoints[12])[1]
            r_hand = get_coords(keypoints[9])
            l_hand = get_coords(keypoints[10])
        except IndexError:
            return

        # Normalize paddle position between shoulder and hip
        try:
            self.paddle1_pos[1] = int(
                round((r_hand[1] - min(r_hand[1], bottom)) / (
                        max(top, r_hand[1]) - min(r_hand[1], bottom)) * self.HEIGHT))
            self.paddle2_pos[1] = int(
                round((l_hand[1] - min(l_hand[1], bottom)) / (
                        max(top, l_hand[1]) - min(l_hand[1], bottom)) * self.HEIGHT))
        except ZeroDivisionError:
            return

    def move_paddle_tp(self, keypoints):
        try:
            if keypoints[0][6][1] < self.WIDTH // 2:
                l_person = keypoints[0]
                r_person = keypoints[1]
            else:
                l_person = keypoints[1]
                r_person = keypoints[0]

            # Get body coords if they exist
            bottom_l = get_coords(l_person[6])[1]
            top_l = get_coords(l_person[12])[1]
            bottom_r = get_coords(r_person[6])[1]
            top_r = get_coords(r_person[12])[1]
            r_hand = get_coords(r_person[9])
            l_hand = get_coords(l_person[10])
        except IndexError:
            print("Two player index error")
            return

        # Normalize paddle position between shoulder and hip
        try:
            self.paddle1_pos[1] = int(
                round((r_hand[1] - min(r_hand[1], bottom_r)) / (
                        max(top_r, r_hand[1]) - min(r_hand[1], bottom_r)) * self.HEIGHT))
            self.paddle2_pos[1] = int(
                round((l_hand[1] - min(l_hand[1], bottom_l)) / (
                        max(top_l, l_hand[1]) - min(l_hand[1], bottom_l)) * self.HEIGHT))
        except ZeroDivisionError:
            return

    def move_paddle(self):
        # Infer image and draw it
        if not self.two_player:
            self.image, keypoints = self.pnet.get_image()
            keypoints = self.scale_keypoints(self.image.shape, keypoints[0])
            self.move_paddle_sp(keypoints)
        else:
            self.image, keypoints = self.pnet.get_image(num_people=2)
            keypoints = self.scale_keypoints(self.image.shape, keypoints)
            self.move_paddle_tp(keypoints)
        # keypoints = self.scale_keypoints(self.image.shape, keypoints)

        # Make sure paddles don't go off screen
        if self.paddle1_pos[1] > self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle1_pos[1] = self.HEIGHT - self.HALF_PAD_HEIGHT
        elif self.paddle1_pos[1] < self.HALF_PAD_HEIGHT:
            self.paddle1_pos[1] = self.HALF_PAD_HEIGHT

        if self.paddle2_pos[1] > self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle2_pos[1] = self.HEIGHT - self.HALF_PAD_HEIGHT
        elif self.paddle2_pos[1] < self.HALF_PAD_HEIGHT:
            self.paddle2_pos[1] = self.HALF_PAD_HEIGHT

    def move_ball(self):
        time_passed = self.clock.get_time() / 1000
        delta_x = self.ball_vel[0] * time_passed
        delta_y = self.ball_vel[1] * time_passed

        new_ball_pos = [0, 0]
        winning_player = None
        # X-Movement
        # Moving Left
        if self.ball_vel[0] < 0:
            # Check if ball will move past paddle
            if self.ball_pos[0] + delta_x - self.BALL_RADIUS <= self.PAD_WIDTH:
                dist_to_paddle = (self.ball_pos[0] - self.BALL_RADIUS) - self.PAD_WIDTH
                time_to_reach_paddle = (-1 * dist_to_paddle) / self.ball_vel[0]
                y_at_paddle = self.ball_pos[1] + self.ball_vel[1] * time_to_reach_paddle
                # Check if it will hit paddle
                if self.paddle1_pos[1] - self.HALF_PAD_HEIGHT <= y_at_paddle <= \
                        self.paddle1_pos[1] + self.HALF_PAD_HEIGHT:
                    movement_after_hit = abs(delta_x) - dist_to_paddle
                    new_ball_pos[0] = movement_after_hit + self.PAD_WIDTH + self.BALL_RADIUS
                    self.ball_vel[0] *= -1.1
                else:
                    winning_player = 'right'
            else:
                new_ball_pos[0] = self.ball_pos[0] + delta_x
        # Moving Right
        elif self.ball_vel[0] > 0:
            # Check if ball will move past paddle
            if self.ball_pos[0] + delta_x + self.BALL_RADIUS >= self.WIDTH - self.PAD_WIDTH:
                dist_to_paddle = (self.WIDTH - self.PAD_WIDTH) - (self.ball_pos[0] + self.BALL_RADIUS)
                time_to_reach_paddle = dist_to_paddle / self.ball_vel[0]
                y_at_paddle = self.ball_pos[1] + self.ball_vel[1] * time_to_reach_paddle
                # Check if it will hit paddle
                if self.paddle2_pos[1] - self.HALF_PAD_HEIGHT <= y_at_paddle <= \
                        self.paddle2_pos[1] + self.HALF_PAD_HEIGHT:
                    movement_after_hit = delta_x - dist_to_paddle
                    new_ball_pos[0] = (self.WIDTH - self.PAD_WIDTH) - movement_after_hit - self.BALL_RADIUS
                    self.ball_vel[0] *= -1.1
                else:
                    winning_player = 'left'
            else:
                new_ball_pos[0] = self.ball_pos[0] + delta_x
        else:
            new_ball_pos[0] = self.ball_pos[0]

        # Y-Movement
        # Moving Up
        if self.ball_vel[1] < 0:
            # Check if ball will move past top of screen
            if self.ball_pos[1] + delta_y - self.BALL_RADIUS <= 0:
                dist_to_top = self.ball_pos[1] - self.BALL_RADIUS
                new_ball_pos[1] = (-1 * delta_y) - dist_to_top + self.BALL_RADIUS
                self.ball_vel[1] *= -1
            else:
                new_ball_pos[1] = self.ball_pos[1] + delta_y
        # Moving Down
        elif self.ball_vel[1] > 0:
            # Check if ball will move past bottom of screen
            if self.ball_pos[1] + delta_x + self.BALL_RADIUS >= self.HEIGHT + 1:
                dist_to_bot = self.HEIGHT - (self.ball_pos[1] + self.BALL_RADIUS)
                new_ball_pos[1] = (self.HEIGHT - self.BALL_RADIUS) - (delta_y - dist_to_bot)
                self.ball_vel[1] *= -1
            else:
                new_ball_pos[1] = self.ball_pos[1] + delta_y

        if winning_player:
            if winning_player == 'left':
                self.l_score += 1
                self.ball_init(True)
            else:
                self.r_score += 1
                self.ball_init(False)
        else:
            self.ball_pos[0] = new_ball_pos[0]
            self.ball_pos[1] = new_ball_pos[1]

    def draw(self):
        # canvas.fill(BLACK)
        if len(enumerate()) == 2:
            thread = Thread(target=self.move_paddle)
            thread.daemon = True
            thread.start()
            # self.frame_count = 0
        
        self.blit_cam_frame(self.image, self.window)
        # self.frame_count += 1

        self.move_ball()

        # Draw paddle lines
        # pygame.draw.line(self.window, WHITE, [WIDTH // 2, 0], [WIDTH // 2, HEIGHT], 1)
        pygame.draw.line(self.window, WHITE, [self.PAD_WIDTH, 0], [self.PAD_WIDTH, self.HEIGHT], 1)
        pygame.draw.line(self.window, WHITE, [self.WIDTH - self.PAD_WIDTH, 0],
                         [self.WIDTH - self.PAD_WIDTH, self.HEIGHT], 1)
        # pygame.draw.circle(self.window, WHITE, [WIDTH // 2, HEIGHT // 2], 70, 1)

        # Draw ball
        pygame.draw.circle(self.window, ORANGE, (int(self.ball_pos[0]), int(self.ball_pos[1])), self.BALL_RADIUS, 0)
        # Draw paddles
        pygame.draw.polygon(self.window, GREEN,
                            [[self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT]],
                            0)
        pygame.draw.polygon(self.window, GREEN,
                            [[self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT]],
                            0)

        # Draw scores and FPS
        myfont1 = pygame.font.SysFont("Comic Sans MS", self.HEIGHT // 10)
        label1 = myfont1.render("Score " + str(self.l_score), 1, (0, 255, 0))
        self.window.blit(label1, (50, 20))

        myfont2 = pygame.font.SysFont("Comic Sans MS", self.HEIGHT // 10)
        label2 = myfont2.render("Score " + str(self.r_score), 1, (0, 255, 0))
        self.window.blit(label2, (self.WIDTH - label2.get_rect().width - self.PAD_WIDTH, 20))
        frames = myfont2.render(str(int(self.clock.get_fps())) + " FPS", True, pygame.Color('white'))
        self.window.blit(frames, (50, 70))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--two_player', dest='two_player', action='store_true')
    parser.set_defaults(two_player=False)
    args = parser.parse_args()

    pong = Pong(args.two_player)
    pong.run()
