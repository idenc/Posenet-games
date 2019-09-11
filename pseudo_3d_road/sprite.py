import pygame


class Sprite:
    def __init__(self, img, height, plane_center, init_x, m, b):
        self.curve_idx = 0
        self.size = height // 90
        self.orig_size = self.size
        self.x = init_x
        self.orig_x = init_x
        self.y = plane_center + 10 + self.size // 2
        self.orig_y = self.y
        self.draw = False
        self.texture_orig = pygame.image.load(img).convert_alpha()
        self.m = m
        self.b = b
        self.speed = None
        self.accel = None
        self.slope = self.get_road_slope()
        self.spawn_time = pygame.time.get_ticks()

    def reset(self, increment, new_x):
        self.size = self.orig_size
        self.x = new_x
        self.orig_x = new_x
        self.y = self.orig_y
        self.draw = False
        self.curve_idx = 0
        self.speed = increment
        self.accel = 2 / (increment * 100)
        self.slope = self.get_road_slope()
        self.spawn_time = pygame.time.get_ticks()

    def get_road_slope(self):
        return self.m * self.orig_x + self.b

    def blit_sprite(self, window, curve, road_start, direction, plane_center, width_scale):
        sprite_rect = None
        if self.draw:
            # Resize sprite
            self.size = int(self.size + self.speed)
            sprite_texture = pygame.transform.scale(self.texture_orig, (self.size * width_scale, self.size))
            # Draw sprite
            sprite_rect = window.blit(sprite_texture,
                                      (self.x - (self.size // 2) +
                                       curve[road_start + self.curve_idx] * direction,
                                       self.y - (self.size // 2)))

            # Move sprite according to perspective
            dx = (self.y - plane_center + 10) * self.slope
            self.x = self.orig_x + dx

            self.y += self.speed  # Move down
            self.speed += self.accel  # Accelerate
            if self.curve_idx != 0:  # Align to curve
                self.curve_idx -= 1

            self.accel += 0.002

        return sprite_rect
