import sys

import pygame
import pygameMenu
import numpy as np
import subprocess

from pygameMenu.locals import *
from pygame.locals import *
from Pong import Pong
from Pong import get_coords
from posenet_interface import posenetInterface
from pynput.keyboard import Key, Controller
from Falling import Dodger
from Gesture import Gesture


class Menu:
    def __init__(self, width=None, height=None):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.frame_count = 2
        self.menu = None
        self.image = None
        self.select_time = pygame.time.get_ticks()
        self.hand_positions = []
        self.hand_times = []
        self.l_hand = [1, 1]
        self.r_hand = [1, 1]

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
        pygame.display.set_caption('Posenet Game Selection')
        self.pnet = posenetInterface(257)
        self.menu_init()
        self.option_boxes = self.menu.get_option_rects()

    def menu_init(self):
        fontdir = pygameMenu.fonts.FONT_8BIT
        self.menu = pygameMenu.Menu(self.window, self.WIDTH, self.HEIGHT, fontdir, "GAME SELECTION",
                                    bgfun=self.draw_bg, fgfun=self.handle_input, dopause=True,
                                    menu_width=self.WIDTH, menu_height=(self.HEIGHT - 50),
                                    font_size_title=self.WIDTH // 16,
                                    option_margin=self.HEIGHT // 10, rect_width=2, font_size=self.HEIGHT // (3 * 5))

        self.menu.add_option('SUPER MARIO BROS', self.med, "SMB")
        self.menu.add_option('TETRIS', self.med, "Tetris")
        self.menu.add_option("Next", self.go_to_menu2)
        self.menu.add_option('Exit', PYGAME_MENU_EXIT)

    def go_to_menu1(self):
        self.menu.reset(1)
        self.menu._size = 0
        del self.menu._actual._option[:]
        self.menu._opt_posx = int(
            self.menu._width * (self.menu._draw_regionx / 100.0)) + self.menu._posy
        self.menu._opt_posy = int(
            self.menu._height * (self.menu._draw_regiony / 100.0)) + self.menu._posx
        self.menu.add_option('SUPER MARIO BROS', self.med, "SMB")
        self.menu.add_option('TETRIS', self.med, "Tetris")
        self.menu.add_option("Next", self.go_to_menu2)
        self.menu.add_option('Exit', PYGAME_MENU_EXIT)
        self.option_boxes = self.menu.get_option_rects()

    def go_to_menu2(self):
        self.menu.reset(1)
        self.menu._size = 0
        del self.menu._actual._option[:]
        self.menu._opt_posx = int(
            self.menu._width * (self.menu._draw_regionx / 100.0)) + self.menu._posy
        self.menu._opt_posy = int(
            self.menu._height * (self.menu._draw_regiony / 100.0)) + self.menu._posx
        self.menu.add_option('PONG', self.pong)
        self.menu.add_option('FALLING ROCKS', self.rocks)
        self.menu.add_option("Back", self.go_to_menu1)
        self.menu.add_option("Next", self.go_to_menu3)
        self.menu.add_option('Exit', PYGAME_MENU_EXIT)
        self.option_boxes = self.menu.get_option_rects()

    def go_to_menu3(self):
        self.menu.reset(1)
        self.menu._size = 0
        del self.menu._actual._option[:]
        self.menu._opt_posx = int(
            self.menu._width * (self.menu._draw_regionx / 100.0)) + self.menu._posy
        self.menu._opt_posy = int(
            self.menu._height * (self.menu._draw_regiony / 100.0)) + self.menu._posx
        self.menu.add_option('EXCITEBIKE', self.med, "Excitebike")
        self.menu.add_option("Back", self.go_to_menu2)
        self.menu.add_option('Exit', PYGAME_MENU_EXIT)
        self.option_boxes = self.menu.get_option_rects()

    def scale_keypoints(self, original_shape, keypoints):
        scale = np.array([self.HEIGHT / original_shape[0], self.WIDTH / original_shape[1]])
        return keypoints * scale

    def draw_bg(self):
        self.window.fill((40, 0, 40))

    def infer_hands(self):
        # Infer image and draw it
        keypoints = self.pnet.get_keypoints()[0]
        keypoints = self.scale_keypoints((self.pnet.cap.get(4), self.pnet.cap.get(3)), keypoints)
        keypoints[:, 1] = self.WIDTH - keypoints[:, 1]
        try:
            # Get body coords if they exist
            bottom = get_coords(keypoints[6])[1]
            top = get_coords(keypoints[12])[1]
            self.r_hand = get_coords(keypoints[10])
            self.l_hand = get_coords(keypoints[9])
        except IndexError:
            return

        # Normalize pointer position between shoulder and hip
        try:
            self.r_hand[1] = int(
                round((self.r_hand[1] - min(self.r_hand[1], bottom)) / (
                        max(top, self.r_hand[1]) - min(self.r_hand[1], bottom)) * self.HEIGHT))
            self.l_hand[1] = int(
                round((self.l_hand[1] - min(self.l_hand[1], bottom)) / (
                        max(top, self.l_hand[1]) - min(self.l_hand[1], bottom)) * self.HEIGHT))
        except ZeroDivisionError:
            return

        self.hand_times.append(pygame.time.get_ticks())
        self.hand_positions.append(self.r_hand)

    def handle_input(self):
        # if len(threading.enumerate()) == 1:
        #     thread = Thread(target=self.infer_hands)
        #     thread.daemon = True
        #     thread.start()
            #self.frame_count = 0
        self.infer_hands()
        #self.frame_count += 1
        pygame.draw.circle(self.window, (255, 0, 0), self.r_hand, 15, 1)
        pygame.draw.circle(self.window, (255, 255, 255), self.l_hand, 15, 1)
        i = 0
        for box in self.option_boxes:
            if box.collidepoint(self.r_hand):
                self.menu.set_option_index(i)

            num_points = self.collide_point_all(box)
            if pygame.time.get_ticks() - self.select_time >= 5000 and num_points > (len(self.hand_positions) * 0.5):
                self.select_time = pygame.time.get_ticks()
                self.menu._select()
            i += 1

        curr_time = pygame.time.get_ticks()
        for i, (time, pos) in enumerate(zip(self.hand_times, self.hand_positions)):
            if curr_time - time >= 5000:
                del self.hand_positions[i]
                del self.hand_times[i]

    def collide_point_all(self, rect):
        count = 0
        for point in self.hand_positions:
            if rect.collidepoint(point[0], point[1]):
                count += 1
        return count

    def run(self):
        while True:
            events = pygame.event.get()
            self.menu.mainloop(events)
            self.clock.tick(60)  # Limit to 60 FPS

    def game_loop(self, game):
        self.menu.disable()
        self.menu.reset(1)

        while True:
            game.draw()
            events = pygame.event.get()
            for event in events:
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.hand_positions = []
                        self.hand_times = []
                        self.select_time = pygame.time.get_ticks()
                        if type(game) == Gesture:
                          flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
                          self.window =   pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=flags)
                          subprocess.call(["pkill", "mednafen"])
                        self.menu.enable()
                        return

            self.menu.mainloop(events)

            pygame.display.update()
            game.clock.tick(60)

    def pong(self):
        pong = Pong(False, pnet=self.pnet)
        self.game_loop(pong)

    def rocks(self):
        rocks = Dodger(3, pnet=self.pnet)
        self.game_loop(rocks)

    def med(self, game):
        keyboard = Controller()
        gesture = Gesture(game, 480, 480, pnet=self.pnet)
        with keyboard.pressed(Key.ctrl):
            with keyboard.pressed(Key.alt):
                keyboard.press(Key.right)
                keyboard.release(Key.right)
        self.game_loop(gesture)

    def mainmenu_background(self):
        """
        Background color of the main menu, on this function user can plot
        images, play sounds, etc.
        """
        self.window.fill((40, 0, 40))


if __name__ == "__main__":
    menu = Menu()
    menu.run()
