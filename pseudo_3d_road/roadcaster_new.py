import pickle
import random
import sys
from threading import Thread, enumerate
from math import *

import numpy as np
import pygame
import pygameMenu
from pygame.locals import *
from sprite import Sprite

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


def check_crouch(keypoints):
    # Get average y position of feet
    foot_pos = []
    for frame in keypoints:
        l_hip_y = frame[5][0]
        r_hip_y = frame[6][0]
        if l_hip_y != 0.0 and r_hip_y != 0.0:
            foot_pos.append((l_hip_y + r_hip_y) / 2)
    if len(foot_pos) > 2:
        low_thresh = max(foot_pos) - 15
        if foot_pos[-1] < low_thresh:
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
        self.road_width = self.texture2.get_width()
        self.road_height = self.texture2.get_height()
        self.player_car = pygame.image.load('player_car.png').convert_alpha()
        self.player_car = pygame.transform.scale(self.player_car, (self.WIDTH // 5, self.HEIGHT // 5))
        self.wall_explode = pygame.image.load('brick_explode.png').convert_alpha()
        self.wall_explode = pygame.transform.scale(self.wall_explode, (self.WIDTH // 4, self.HEIGHT // 4))
        ground = pygame.Surface((640, 240)).convert()
        ground.fill((0, 100, 0))
        self.resolution = self.WIDTH // 640

        self.bright_image = self.player_car.copy()
        self.bright_image.fill((200, 200, 200, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # field of view (FOV)
        self.fov = 60

        self.player_height = 64 / 2
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

        # Wall objects
        m, b = self.get_road_eqn()
        self.increment = None
        self.wall1 = Sprite('wall.png', self.HEIGHT, self.plane_center, self.get_rand_x(), m, b)
        self.wall2 = Sprite('wall.png', self.HEIGHT, self.plane_center, self.get_rand_x(), m, b)
        self.wall_dist = None

        # Star objects
        self.star = Sprite('star.png', self.HEIGHT, self.plane_center, self.get_rand_x(), m, b)
        self.star_texture = pygame.transform.scale(pygame.image.load('star.png').convert_alpha(),
                                                   (self.star.texture_orig.get_width() // 2,
                                                    self.star.texture_orig.get_height() // 2))

        # Timer fonts
        self.start_time = pygame.time.get_ticks()
        self.time_font = pygame.font.SysFont(pygameMenu.fonts.FONT_8BIT, self.HEIGHT // 20)

        # Player status
        self.score = 0
        self.jump = False
        self.car_vel = self.HEIGHT // 1.4
        self.has_star = False
        self.star_active = False
        self.star_activate_time = 0
        self.flicker_time = 0
        self.bright = False
        self.collided_wall = None
        self.draw_wall_break = False
        self.wall_break_time = 0

        # Gesture detection
        self.keypoints = []
        self.kp_time = pygame.time.get_ticks()

    def get_rand_x(self):
        return random.randint(self.plane_center_x - (self.WIDTH // 96),
                              self.plane_center_x + (self.WIDTH // 96))

    def get_road_eqn(self):
        # Desired slope range is -2.35 to 2.35 => dist is 4.7
        m = 4.7 / ((self.plane_center_x + (self.WIDTH // 96)) - (self.plane_center_x - (self.WIDTH // 96)))
        b = -2.35 - m * (self.plane_center_x - (self.WIDTH // 96))
        return m, b

    def reset_wall(self):
        self.wall1.reset(self.increment, self.get_rand_x())
        self.wall2.reset(self.increment, self.get_rand_x())
        self.wall_dist = None

    def move_car(self):
        keypoints = self.pnet.get_keypoints()
        keypoints = self.scale_keypoints((self.pnet.cap.get(3), self.pnet.cap.get(4)), keypoints[0])
        self.keypoints.append(keypoints)
        if keypoints[11][1] != 0.0:
            # Adjust car position to player position
            self.car_pos[0] = self.WIDTH - int(keypoints[11][1])
        if not self.jump and check_jump(keypoints):
            self.jump = True

        if pygame.time.get_ticks() - self.kp_time >= 400:
            if self.has_star and check_crouch(self.keypoints):
                self.star_active = True
                self.star_activate_time = pygame.time.get_ticks()
                self.has_star = False
            del self.keypoints[:len(self.keypoints) - 1]
            self.kp_time = pygame.time.get_ticks()

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
                    elif event.key == K_LEFT:
                        self.car_pos[0] -= 20
                    elif event.key == K_RIGHT:
                        self.car_pos[0] += 20
                    elif event.key == K_SPACE:
                        self.jump = True
                    elif event.key == K_f and self.has_star:
                        self.star_active = True
                        self.has_star = False

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
        if pygame.time.get_ticks() - self.wall1.spawn_time >= 1000 - (250 * self.score) and not self.wall1.draw \
                and not self.draw_wall_break:
            self.wall1.draw = True
            self.wall2.draw = True
            self.wall1.curve_idx = road_index
            self.wall2.curve_idx = road_index
            if self.increment is None:
                self.increment = int(60 // self.clock.get_fps())  # How much to change wall based on FPS
                self.wall1.speed = self.increment
                self.wall1.accel = 2 / (self.increment * 100)
                self.wall2.speed = self.increment
                self.wall2.accel = 2 / (self.increment * 100)

        # Spawn star power up
        if pygame.time.get_ticks() - self.star.spawn_time >= 3000 and not self.star.draw:
            self.star.draw = True
            self.star.curve_idx = road_index
            self.star.speed = self.increment
            self.star.accel = 2 / (self.increment * 100)

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

    def draw_sprites(self):
        star_rect = self.star.blit_sprite(
            self.window, self.curve, self.road_start, self.direction, self.plane_center, 1)

        car_img = self.player_car
        if self.star_active:
            if pygame.time.get_ticks() - self.flicker_time >= 100:
                self.bright = not self.bright
                self.flicker_time = pygame.time.get_ticks()
            if self.bright:
                car_img = self.bright_image

        if self.draw_wall_break:
            self.window.blit(self.wall_explode, self.collided_wall)
            if pygame.time.get_ticks() - self.wall_break_time >= 200:
                self.draw_wall_break = False

        # Whether to draw wall behind car
        if self.wall1.y + self.wall1.size // 2 >= self.HEIGHT:
            car_rect = self.window.blit(car_img, self.car_pos)
            wall_rect1 = self.wall1.blit_sprite(
                self.window, self.curve, self.road_start, self.direction, self.plane_center, 2)
            wall_rect2 = self.wall2.blit_sprite(
                self.window, self.curve, self.road_start, self.direction, self.plane_center, 2)
        else:
            wall_rect1 = self.wall1.blit_sprite(
                self.window, self.curve, self.road_start, self.direction, self.plane_center, 2)
            wall_rect2 = self.wall2.blit_sprite(
                self.window, self.curve, self.road_start, self.direction, self.plane_center, 2)
            car_rect = self.window.blit(car_img, self.car_pos)

        if wall_rect1 is not None and wall_rect1.top >= self.HEIGHT:  # Wall is at bottom of screen
            self.wall1.reset(self.increment, self.get_rand_x())
            self.score += 1

        if wall_rect2 is not None and wall_rect2.top >= self.HEIGHT:
            self.wall2.reset(self.increment, self.get_rand_x())

        if self.star.y >= self.HEIGHT:
            self.star.reset(self.increment, self.get_rand_x())

        self.check_collide(car_rect, wall_rect1, wall_rect2)

        if not self.jump and star_rect and car_rect.colliderect(star_rect):
            self.has_star = True
            self.star.reset(self.increment, self.get_rand_x())

        if self.has_star:
            self.window.blit(self.star_texture, (50, self.plane_center + 10))

    def player_fail(self):
        time = pygame.time.get_ticks()
        self.start_time = time
        self.failed = True
        self.fail_time = time
        self.score = 0

    def check_collide(self, car_rect, wall_rect1, wall_rect2):
        if not self.jump:
            # Check if car and wall collide
            collided = False
            if wall_rect1 is not None and wall_rect1.bottom < self.HEIGHT and car_rect.colliderect(wall_rect1):
                self.collided_wall = (wall_rect1.left, wall_rect1.top)
                collided = True
                self.wall1.reset(self.increment, self.get_rand_x())
            elif wall_rect2 is not None and wall_rect2.bottom < self.HEIGHT and car_rect.colliderect(wall_rect2):
                self.collided_wall = (wall_rect2.left, wall_rect2.top)
                collided = True
                self.wall2.reset(self.increment, self.get_rand_x())
            if collided:
                if self.star_active:
                    self.draw_wall_break = True
                    self.wall_break_time = pygame.time.get_ticks()
                else:
                    self.player_fail()

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

        if self.jump and not self.failed:
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
        self.draw_sprites()
        self.draw_labels()

        if pygame.time.get_ticks() - self.star_activate_time >= 10000:
            self.star_activate_time = pygame.time.get_ticks()
            self.star_active = False

        # measure the framerate
        # print(self.clock.get_fps())
        pygame.display.flip()


if __name__ == "__main__":
    rc = RoadCaster(720, 480)
    rc.run()
