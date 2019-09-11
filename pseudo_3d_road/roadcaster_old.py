import pickle
import random
import sys
from threading import Thread, enumerate
from math import *

import numpy as np
import pygame
import pygameMenu
from pygame.locals import *

sys.path.append("..")
import posenet_interface

BLACK = pygame.color.THECOLORS["black"]
WHITE = pygame.color.THECOLORS["white"]
BLUE = (0, 0, 255)
GREY = (245, 245, 245)


def check_jump(keypoints):
    l_hand_y = keypoints[9][0]
    r_hand_y = keypoints[10][0]
    l_elbow_y = keypoints[7][0]
    r_elbow_y = keypoints[8][0]
    if l_hand_y != 0.0 and r_hand_y != 0.0 and l_hand_y < l_elbow_y and r_hand_y < r_elbow_y:
        print("Jump")
        return True
    return False


class RoadCaster:
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
        pygame.display.set_caption('Posenet Roadcaster')

        if pnet:
            self.pnet = pnet
        else:
            self.pnet = posenet_interface.posenetInterface(257)

        # Display objects
        self.plane_center_x = self.WIDTH // 2
        self.texture = pygame.image.load('stripes.png').convert()
        self.texture = pygame.transform.scale(self.texture, (self.WIDTH, self.texture.get_height()))
        self.texture2 = pygame.image.load('road3.png').convert()
        self.wall_texture_orig = pygame.image.load('wall.png').convert()
        self.road_width = self.texture2.get_width()
        self.road_height = self.texture2.get_height()
        self.player_car = pygame.image.load('player_car.png').convert_alpha()
        self.player_car = pygame.transform.scale(self.player_car, (self.WIDTH // 5, self.HEIGHT // 5))
        ground = pygame.Surface((640, 240)).convert()
        ground.fill((0, 100, 0))
        self.resolution = self.WIDTH // 640

        self.bright_image = self.player_car.copy()
        self.bright_image.fill((200, 200, 200, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # field of view (FOV)
        self.fov = 60
        wall_height = 64

        self.player_height = wall_height / 2
        self.player_pos = [self.WIDTH // 4, int((self.WIDTH // 4) * 1.4)]
        self.car_pos = [(self.WIDTH // 2), (self.WIDTH // 2) - 80]
        view_angle = 90

        # Center of the Projection Plane
        self.plane_center = self.HEIGHT // 2  # [WIDTH/2, HEIGHT/2]
        # distance from player to projection plane
        self.to_plane_dist = int((self.WIDTH / 2) / tan(radians(self.fov / 2)))

        # angle of the casted ray
        ray_angle = 90  # view_angle+(fov/2)
        beta = radians(view_angle - ray_angle)
        self.cos_beta = cos(beta)
        self.sin_angle = -sin(radians(ray_angle))

        move_speed = 15
        self.x_move = int(move_speed * cos(radians(view_angle)))
        self.y_move = -int(move_speed * sin(radians(view_angle)))

        with open('default.road', 'rb') as file:
            file_loader = pickle.Unpickler(file)
            self.curve = file_loader.load()

        self.road_start = 0  # this indicate the position on the road
        self.direction = 1  # this is used to change the curve direction
        self.road_index = 0  # this is used to index all points on the curve
        self.speed = 1  # this is used to increment road_start

        # Wall objects
        self.m, self.b = self.get_road_eqn()
        self.wall_time = pygame.time.get_ticks()
        self.wall_idx = 0
        self.wall_size = self.HEIGHT // 90
        self.wall_x = self.get_wall_x()
        self.orig_x1 = self.wall_x
        self.wall2_x = self.get_wall_x()
        self.orig_x2 = self.wall2_x
        self.slope_wall1 = self.get_wall_slope(self.wall_x)
        self.slope_wall2 = self.get_wall_slope(self.wall2_x)
        self.wall_y = self.plane_center + 10 + self.wall_size // 2
        self.draw_wall = False
        self.increment = None
        self.wall_speed = None
        self.wall_accel = None

        # Timer fonts
        self.start_time = pygame.time.get_ticks()
        self.time_font = pygame.font.SysFont(pygameMenu.fonts.FONT_8BIT, self.HEIGHT // 20)

        # Fail objects
        fail_font = pygame.font.SysFont(pygameMenu.fonts.FONT_8BIT, self.HEIGHT // 10)
        self.fail_label = fail_font.render("FAILED", 1, (255, 0, 0))
        self.fail_rect = self.fail_label.get_rect(center=(
            self.WIDTH // 2, (self.HEIGHT // 2) + self.fail_label.get_height()))
        self.failed = False
        self.fail_time = 0
        self.explosion = pygame.image.load('Explosion.png').convert()
        explosion_scale = self.player_car.get_width() // self.explosion.get_height()
        self.explosion = pygame.transform.scale(self.explosion,
                                                (self.explosion.get_width() * explosion_scale,
                                                 self.explosion.get_height() * explosion_scale))
        self.explosion_idx = 1

        self.score = 0
        self.jump = False
        self.car_vel = self.HEIGHT // 1.3

    def get_wall_x(self):
        return random.randint(self.plane_center_x - (self.WIDTH // 96),
                              self.plane_center_x + (self.WIDTH // 96))

    def get_road_eqn(self):
        # Desired slope range is -2.35 to 2.35 => dist is 4.7
        m = 4.7 / ((self.plane_center_x + (self.WIDTH // 96)) - (self.plane_center_x - (self.WIDTH // 96)))
        b = -2.35 - m * (self.plane_center_x - (self.WIDTH // 96))
        return m, b

    def get_wall_slope(self, x):
        return self.m * x + self.b

    def reset_wall(self):
        self.wall_speed = self.increment
        self.wall_size = self.HEIGHT // 90
        self.wall_x = self.get_wall_x()
        self.slope_wall1 = self.get_wall_slope(self.wall_x)
        self.wall2_x = self.get_wall_x()
        self.slope_wall2 = self.get_wall_slope(self.wall2_x)
        self.wall_y = self.plane_center + 10 + self.wall_size // 2
        self.draw_wall = False
        self.wall_idx = 0
        self.wall_speed = self.increment
        self.wall_accel = 2 / (self.increment * 100)
        self.wall_time = pygame.time.get_ticks()

    def move_car(self):
        keypoints = self.pnet.get_keypoints()
        keypoints = self.scale_keypoints((self.pnet.cap.get(3), self.pnet.cap.get(4)), keypoints[0])
        if keypoints[11][1] != 0.0:
            # Adjust car position to player position
            self.car_pos[0] = self.WIDTH - int(keypoints[11][1])
        if not self.jump and check_jump(keypoints):
            self.jump = True

    def run(self):
        self.window.fill(BLUE)
        while 1:
            self.draw()
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            # pygame.display.update()
            self.clock.tick(30)

    def scale_keypoints(self, original_shape, keypoints):
        scale = np.array([self.HEIGHT / original_shape[0], self.WIDTH / original_shape[1]])
        return keypoints * scale

    def blit_road(self):
        # now floor-casting and ceilings
        wall_bottom = self.HEIGHT
        road_index = 0
        # Draw road
        while wall_bottom > self.plane_center + 10:
            wall_bottom -= self.resolution
            # (row at floor point-row of center)
            row = wall_bottom - self.plane_center
            # straight distance from player to the intersection with the floor
            straight_p_dist = (self.player_height / row * self.to_plane_dist)
            # true distance from player to floor
            to_floor_dist = (straight_p_dist / self.cos_beta)
            # coordinates (x,y) of the floor
            ray_y = int(self.player_pos[1] + (to_floor_dist * self.sin_angle))
            # the texture position
            floor_y = (ray_y % self.road_height)
            slice_width = int(self.road_width / to_floor_dist * self.to_plane_dist)
            slice_x = self.plane_center_x - (slice_width // 2)
            row_slice = self.texture2.subsurface(0, floor_y, self.road_width, 1)
            row_slice = pygame.transform.scale(row_slice, (slice_width, self.resolution))
            self.window.blit(self.texture, (0, wall_bottom), (0, floor_y, self.WIDTH, self.resolution))
            self.window.blit(row_slice,
                             (slice_x + self.curve[self.road_start + road_index] * self.direction, wall_bottom))
            road_index += 1

        # Spawn walls faster as score increases
        if pygame.time.get_ticks() - self.wall_time >= 1000 - (250 * self.score) and not self.draw_wall:
            self.draw_wall = True
            self.wall_idx = road_index

    def blit_wall(self):
        wall_rect1 = None
        wall_rect2 = None
        if self.draw_wall:
            # self.increment = 1
            if self.increment is None:
                self.increment = int(60 // self.clock.get_fps())  # How much to change wall based on FPS
                self.wall_speed = self.increment
                self.wall_accel = 2 / (self.increment * 100)

            # Resize walls
            self.wall_size = int(self.wall_size + self.wall_speed)
            wall_texture = pygame.transform.scale(self.wall_texture_orig, (self.wall_size * 2, self.wall_size))
            # Draw walls
            wall_rect1 = self.window.blit(wall_texture,
                                          (self.wall_x - (self.wall_size // 2) +
                                           self.curve[self.road_start + self.wall_idx] * self.direction,
                                           self.wall_y - (self.wall_size // 2)))
            wall_rect2 = self.window.blit(wall_texture,
                                          (self.wall2_x - (self.wall_size // 2) +
                                           self.curve[self.road_start + self.wall_idx] * self.direction,
                                           self.wall_y - (self.wall_size // 2)))

            # Move wall according to perspective
            dx1 = (self.wall_y - self.plane_center + 10) * self.slope_wall1
            dx2 = (self.wall_y - self.plane_center + 10) * self.slope_wall2
            self.wall_x = self.orig_x1 + dx1
            self.wall2_x = self.orig_x2 + dx2

            self.wall_y += self.wall_speed  # Move down
            self.wall_speed += self.wall_accel  # Accelerate
            if self.wall_idx != 0:  # Align to curve
                self.wall_idx -= 1
            if self.wall_y >= self.HEIGHT:  # Wall is at bottom of screen
                self.reset_wall()
                self.score += 1
            self.wall_accel += 0.002

        return wall_rect1, wall_rect2

    def draw_labels(self):
        # Draw time passed since failed
        time = round((pygame.time.get_ticks() - self.start_time) / 1000, 1)
        label1 = self.time_font.render("Time: " + str(time), 1, (0, 0, 0))
        self.window.blit(label1, (50, self.HEIGHT - self.time_font.get_height()))

        if self.failed:
            # Draw Failed text and explosion animation
            self.window.blit(self.fail_label, self.fail_rect)
            size = self.explosion.get_height()
            self.window.blit(self.explosion,
                             (self.car_pos[0] + (self.player_car.get_width() // 2) - (size // 2),
                              self.car_pos[1] - self.player_car.get_height() + (size // 2)),
                             (size * self.explosion_idx, 0, size, size))

            time_passed = pygame.time.get_ticks() - self.fail_time
            if time_passed >= 100 * self.explosion_idx:
                self.explosion_idx += 1
            if time_passed >= 1200:
                self.failed = False
                self.explosion_idx = 1

    def draw_wall_and_car(self):
        # Whether to draw wall behind car
        if self.wall_y + self.wall_size // 2 >= self.HEIGHT:
            car_rect = self.window.blit(self.player_car, self.car_pos)
            wall_rect1, wall_rect2 = self.blit_wall()
        else:
            wall_rect1, wall_rect2 = self.blit_wall()
            car_rect = self.window.blit(self.player_car, self.car_pos)

        if not self.jump and wall_rect1 is not None and \
                wall_rect1.top < car_rect.bottom and \
                (car_rect.colliderect(wall_rect1) or
                 car_rect.colliderect(wall_rect2)):  # Check if car and wall collide
            time = pygame.time.get_ticks()
            self.start_time = time
            self.failed = True
            self.fail_time = time
            self.reset_wall()
            self.score = 0

    def jump_car(self):
        time_passed = self.clock.get_time() / 1000
        delta_y = self.car_vel * time_passed
        self.car_pos[1] -= delta_y
        self.car_vel -= self.HEIGHT // 25  # subtract gravity
        if self.car_pos[1] > (self.WIDTH // 2) - 80:
            self.car_pos[1] = (self.WIDTH // 2) - 80
            self.car_vel = self.HEIGHT // 1.3
            self.jump = False

    def move_forward(self):
        # Movement controls
        self.player_pos[0] += self.x_move
        self.player_pos[1] += self.y_move

        if self.jump:
            self.jump_car()

        if self.player_pos[1] < 0:
            self.player_pos[1] = 5000
        self.road_start += self.speed

        # if we increment road_start and reach the end of the road
        if self.road_start >= len(self.curve) - len(self.curve) // 2:
            self.road_start = len(self.curve) - len(self.curve) // 2
            # we reverse the incrementation of road_start to exit the curve
            self.speed *= -1
        # if we decrement road_start to exit the curve and reach the start of the road
        elif self.road_start <= 0:
            self.road_start = 0
            # we reverse the incrementation of road_start to start the curve from begining
            self.speed *= -1
            # we reverse the direction of the curve to change the way
            self.direction *= -1

    def draw(self):
        if len(enumerate()) == 2:
            Thread(target=self.move_car, daemon=True).start()

        self.move_forward()
        self.blit_road()
        self.draw_wall_and_car()
        self.draw_labels()

        # measure the framerate
        # print(self.clock.get_fps())
        pygame.display.flip()


if __name__ == "__main__":
    rc = RoadCaster()
    rc.run()
