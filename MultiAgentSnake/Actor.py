from abc import ABC, abstractmethod
import numpy as np
from pyglet.window import key

from pyglet.window.key import KeyStateHandler


class Actor(ABC):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @abstractmethod
    def choose_action(self, game_state: np.ndarray) -> str:
        pass


class ArrowKeyboardActor(Actor):
    def __init__(self, name, key_state_handler: KeyStateHandler):
        super().__init__(name)
        self.last_action = 'up'
        self.key_state_handler = key_state_handler

    def choose_action(self, game_state: np.ndarray) -> str:
        if self.key_state_handler[key.LEFT]:
            self.last_action = 'left'
        elif self.key_state_handler[key.RIGHT]:
            self.last_action = 'right'
        elif self.key_state_handler[key.UP]:
            self.last_action = 'up'
        elif self.key_state_handler[key.DOWN]:
            self.last_action = 'down'
        return self.last_action

    def on_key_press(self, symbol, modifiers):
        if symbol == key.LEFT:
            self.last_action = 'left'
        elif symbol == key.RIGHT:
            self.last_action = 'right'
        elif symbol == key.UP:
            self.last_action = 'up'
        elif symbol == key.DOWN:
            self.last_action = 'down'


class WSADKeyboardActor(Actor):
    def __init__(self, name, key_state_handler: KeyStateHandler):
        super().__init__(name)
        self.last_action = 'down'
        self.key_state_handler = key_state_handler

    def choose_action(self, game_state: np.ndarray) -> str:
        if self.key_state_handler[key.A]:
            self.last_action = 'left'
        elif self.key_state_handler[key.D]:
            self.last_action = 'right'
        elif self.key_state_handler[key.W]:
            self.last_action = 'up'
        elif self.key_state_handler[key.S]:
            self.last_action = 'down'
        return self.last_action

    def on_key_press(self, symbol, modifiers):
        if symbol == key.A:
            self.last_action = 'left'
        elif symbol == key.D:
            self.last_action = 'right'
        elif symbol == key.W:
            self.last_action = 'up'
        elif symbol == key.S:
            self.last_action = 'down'
