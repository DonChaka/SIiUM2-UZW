from __future__ import annotations

import pickle
import random
from itertools import product
from random import randint
from typing import NoReturn, Optional
from time import perf_counter, perf_counter_ns

import pyglet
from pyglet import image
from pyglet.graphics import Batch
from pyglet.shapes import Rectangle
import numpy as np
from numpy import ndarray

from pyglet.window import key, Window

from Actor import Actor


class Point:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point({self.x}, {self.y})'

    def __str__(self):
        return f'Point({self.x}, {self.y})'

    def __add__(self, other) -> Point:
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        assert len(other) == 2
        return Point(self.x + other[0], self.y + other[1])

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

    def out_of_bounds(self, x, y):
        return not (0 <= self.x < x and 0 <= self.y <= y)

    def serialize(self):
        return [self.x, self.y]


class Snake:
    def __init__(self, x, y, color=(0, 0, 255), body: Optional[ndarray] = None):
        self.head: Point = Point(x, y)
        self.body: list[Point] = [Point(body_part[0], body_part[1]) for body_part in body] if body is not None else []
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

    def serialize(self):
        ret = [self.head.serialize()]
        for b in self.body:
            ret.append(b.serialize())
        return ret


class GameState:
    __directions = {
        'down': Point(0, 1),
        'up': Point(0, -1),
        'left': Point(-1, 0),
        'right': Point(1, 0),
    }

    __game_states_dir = 'gamestates'
    __apple_value = 10

    def __init__(self, x_size: int, y_size: int, cell_size: int = 40, padding: int = 5):
        self.x_size = x_size
        self.y_size = y_size

        self.snakes: list[Snake] = []
        self.dead_snakes: list[int] = []
        self.apple: Point = Point(0, 0)
        self.score = 0

        self._cell_size = cell_size
        self._padding = padding
        self._garbage = []

    def add_player(self, x: int, y: int, body: Optional[ndarray]) -> NoReturn:
        color = (randint(0, 255), randint(0, 255), randint(0, 255)) if self.snakes else (0, 0, 255)
        self.snakes.append(Snake(x, y, color, body))
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
            self.snakes[actor_index].move(self.__directions[direction])
            self._generate_apple()
            self.score += -10 if actor_index else 10
        else:
            self.snakes[actor_index].move(self.__directions[direction])

    def get_all_states(self) -> list:
        try:
            states = self.__all_states
        except AttributeError:
            states = self._generate_all_possible_states(len(self.snakes[0].body) + 1)
            self.__all_states = states
        return states

    def is_terminal(self):
        return len(self.dead_snakes)

    def get_possible_actions(self, state, actor_index=0):
        pos_actions = []
        snake = state[3][actor_index]
        for action, direction in self.__directions.items():
            next_pos = direction + snake[0]
            if not [next_pos.x, next_pos.y] == snake[1]:
                pos_actions.append(action)
        return tuple(pos_actions)

    def get_next_states(self, state, action, actor_index=0):
        _states = []
        actor_snake = state[3][actor_index].copy()
        other_snake = state[3][(actor_index + 1) % len(self.snakes)].copy()
        target: Point = self.__directions[action] + actor_snake[0]
        if target == Point(state[2][0], state[2][1]):
            apple_changed = True
        if not (target.out_of_bounds(state[0], state[1]) or [target.x, target.y] in other_snake):
            actor_snake.insert(0, [target.x, target.y])
            actor_snake.pop()

        for _action in self.get_possible_actions(state, actor_index=(actor_index+1) % len(self.snakes)):
            apple_changed = False
            temp_snake = other_snake.copy()
            target = self.__directions[_action] + temp_snake[0]
            if target == Point(state[2][0], state[2][1]):
                apple_changed = True

            if not (target.out_of_bounds(state[0], state[1]) or [target.x, target.y] in actor_snake):
                temp_snake.insert(0, [target.x, target.y])
                temp_snake.pop()

            snakes = [temp_snake, actor_snake] if actor_index else [actor_snake, temp_snake]

            if not apple_changed:
                _states.append([state[0], state[1], state[2], snakes])
            else:
                for possible_state in self.get_all_states():
                    if possible_state[3] == snakes and possible_state[2] is not state[2]:
                        _states.append(state)
        return _states

    def _out_of_bounds(self, xy):
        x, y = xy[0], xy[1]
        return not (0 <= x < self.x_size and 0 <= y < self.y_size)

    def _generate_all_possible_states(self, player_length) -> list:
        states = []
        for x1, y1 in product(range(self.x_size), range(self.y_size)):
            p1 = [[x1, y1]]
            for _ in range(player_length - 1):
                for dx1, dy1 in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                    next_cell1 = [p1[-1][0] + dx1, p1[-1][1] + dy1]
                    if next_cell1 in p1 or self._out_of_bounds(next_cell1):
                        continue
                    else:
                        p1.append(next_cell1)
                        break

            for x2, y2 in product(range(self.x_size), range(self.y_size)):
                if [x2, y2] in p1:
                    continue
                p2 = [[x2, y2]]
                for _ in range(player_length - 1):
                    for dx2, dy2 in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                        next_cell2 = [p2[-1][0] + dx2, p2[-1][1] + dy2]
                        if next_cell2 in p2 or next_cell2 in p1 or self._out_of_bounds(next_cell2):
                            continue
                        else:
                            p2.append(next_cell2)
                            break
                if len(p2) != player_length:
                    continue

                for x3, y3 in product(range(self.x_size), range(self.y_size)):
                    fruit = [x3, y3]
                    if fruit in p1 or fruit in p2:
                        continue
                    state = [self.x_size, self.y_size, fruit, [p1, p2]]
                    states.append(state)
        return states

    def state(self):
        snakes = [snake.serialize() for snake in self.snakes]
        return [self.x_size, self.y_size, self.apple.serialize(), snakes]


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


class Game:
    def __init__(self, x_size: int, y_size: int, cell_size: int = 40, padding: int = 5, window: Window = None):
        self._state = GameState(x_size, y_size, cell_size, padding)
        self.window = window
        self.actors: list[Actor] = []

    def add_player(self, x: int, y: int, body, actor: Actor) -> NoReturn:
        self._state.add_player(x, y, body)
        self.actors.append(actor)

    @Window.event
    def on_draw(self):
        self.window.clear()
        batch = Batch()
        self._state.draw(batch)

    def update(self, dt):
        for i, actor in enumerate(self.actors):
            self._state.move(actor.choose_action(self._state.get_state()), i)

# SIZE_X = 8
# SIZE_Y = 8
# SQUARE_SIZE = 40
# PADDING = 3
#
# board = GameState(SIZE_X, SIZE_Y, SQUARE_SIZE, PADDING)
# board.add_player(0, 4, np.array([[0, 5], [0, 6], [0, 7]]))
# board.add_player(5, 4, np.array([[5, 5], [5, 6], [5, 7]]))
#
# all_states = board.get_all_states()
# example_state = all_states[41076]
# print(example_state)
# _states = board.get_next_states(example_state, 'down')

