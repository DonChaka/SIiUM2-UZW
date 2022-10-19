from __future__ import annotations

import random
from random import randint
from typing import NoReturn

import pyglet
from pyglet import image
from pyglet.graphics import Batch
from pyglet.shapes import Rectangle
import numpy as np

from pyglet.window import key, Window

from Actor import Actor


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other: Point) -> Point:
        assert isinstance(other, Point)
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        assert isinstance(other, Point)
        return Point(self.x - other.x, self.y - other.y)

    def __copy__(self) -> Point:
        return Point(self.x, self.y)

    def __contains__(self, item):
        return self.x == item.x and self.y == item.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def copy(self) -> Point:
        return Point(self.x, self.y)


class Snake:
    def __init__(self, x, y, color=(0, 0, 255)):
        self.head: Point = Point(x, y)
        self.body: list[Point] = []
        self._cell_size = 40
        self.color = color

    def move(self, direction: Point, ate: bool = False) -> NoReturn:
        self.body.insert(0, self.head.copy())
        self.head += direction
        if not ate:
            self.body.pop()

    def collides_with_other(self, other: Snake) -> bool:
        return self.head == other.head or self.head in other.body or other.head in self.body

    def colides_with_self(self) -> bool:
        return self.head in self.body

    def collides_with_point(self, point: Point) -> bool:
        return point in self.body or self.head == point


class GameState:
    __directions = {
        'up': Point(0, 1),
        'down': Point(0, -1),
        'left': Point(-1, 0),
        'right': Point(1, 0),
    }

    def __init__(self, x_size: int, y_size: int, cell_size: int = 40, padding: int = 5):
        self.x_size = x_size
        self.y_size = y_size

        self.snakes: list[Snake] = []
        self.dead_snakes: list[int] = []
        self.apple: Point = Point(0, 0)

        self._cell_size = cell_size
        self._padding = padding
        self._garbage = []

    def add_player(self, x: int, y: int) -> NoReturn:
        color = (randint(0, 255), randint(0, 255), randint(0, 255)) if self.snakes else (0, 0, 255)
        self.snakes.append(Snake(x, y, color))
        self._generate_apple()

    def move(self, direction: str, actor_index: int = 0) -> NoReturn:
        if actor_index in self.dead_snakes:
            return
        target = self.snakes[actor_index].head + self.__directions[direction]
        if target.x < 0 or target.x >= self.x_size or target.y < 0 or target.y >= self.y_size:
            self.dead_snakes.append(actor_index)
            return
        for snake in self.snakes:
            if snake.collides_with_point(target):
                self.dead_snakes.append(actor_index)
                return

        if target == self.apple:
            self.snakes[actor_index].move(self.__directions[direction], ate=True)
            self._generate_apple()
        else:
            self.snakes[actor_index].move(self.__directions[direction])

    def _generate_apple(self) -> NoReturn:
        possible = np.transpose(np.where(self.get_state() == 0))
        chosen = random.choice(possible)
        self.apple = Point(chosen[1], chosen[0])

    def get_state(self) -> np.ndarray:
        state = np.zeros((self.x_size, self.y_size))
        for i, snake in enumerate(self.snakes):
            state[snake.head.x, snake.head.y] = i * 2 + 1
            for body in snake.body:
                state[body.x, body.y] = i * 2 + 2

        state[self.apple.x, self.apple.y] = -1
        return np.transpose(state)

    def draw(self, batch: Batch = None):
        self._garbage = []
        for snake in self.snakes:
            self._garbage.append(
                Rectangle(snake.head.x * self._cell_size + self._padding,
                          snake.head.y * self._cell_size + self._padding,
                          self._cell_size - 2 * self._padding,
                          self._cell_size - 2 * self._padding,
                          color=snake.color, batch=batch))
            for body in snake.body:
                self._garbage.append(
                    Rectangle(body.x * self._cell_size + self._padding,
                              body.y * self._cell_size + self._padding,
                              self._cell_size - 2 * self._padding,
                              self._cell_size - 2 * self._padding,
                              color=snake.color, batch=batch))
                self._garbage[-1].opacity = 100

        self._garbage.append(
            Rectangle(self.apple.x * self._cell_size + self._padding,
                      self.apple.y * self._cell_size + self._padding,
                      self._cell_size - 2 * self._padding,
                      self._cell_size - 2 * self._padding,
                      color=(0, 255, 0), batch=batch))


class Game():
    def __init__(self, x_size: int, y_size: int, cell_size: int = 40, padding: int = 5, window: Window = None):
        self._state = GameState(x_size, y_size, cell_size, padding)
        self.window = window
        self.actors: list[Actor] = []

    def add_player(self, x: int, y: int, actor: Actor) -> NoReturn:
        self._state.add_player(x, y)
        self.actors.append(actor)

    @Window.event
    def on_draw(self):
        self.window.clear()
        batch = Batch()
        self._state.draw(batch)

    def update(self, dt):
        for i, actor in enumerate(self.actors):
            self._state.move(actor.choose_action(self._state.get_state()), i)

# cell_size = 40
# padding = 2
# width = 5
# height = 5
#
# window = pyglet.window.Window(width=width * cell_size, height=height * cell_size)
# batch = Batch()
#
# game = GameState(width, height, cell_size, padding)
# game.add_player(width // 2, height // 2)
#
#
# @window.event
# def on_draw():
#     window.clear()
#     game.draw(batch=batch)
#     batch.draw()
#
#
# def update():
#     game.move('right')
#     game.draw(batch=batch)
#     batch.draw()
#
#
# @window.event
# def on_key_press(symbol, modifiers):
#     if symbol == key.LEFT:
#         game.move('left')
#     elif symbol == key.RIGHT:
#         game.move('right')
#     elif symbol == key.UP:
#         game.move('up')
#     elif symbol == key.DOWN:
#         game.move('down')
#
#
# pyglet.app.run()
