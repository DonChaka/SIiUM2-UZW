import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from pyglet.window import key
from game import Point

from pyglet.window.key import KeyStateHandler

from pickle import dump, load


class Actor(ABC):

    def __init__(self, name):
        self.name = name
        self._action_space = ['up', 'down', 'left', 'right']

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @abstractmethod
    def choose_action(self, game_state: list) -> str:
        pass


class ArrowKeyboardActor(Actor):
    def __init__(self, name, key_state_handler: KeyStateHandler):
        super().__init__(name)
        self.last_action = 'up'
        self.key_state_handler = key_state_handler

    def choose_action(self, game_state: list) -> str:
        if self.key_state_handler[key.LEFT]:
            self.last_action = 'left'
        elif self.key_state_handler[key.RIGHT]:
            self.last_action = 'right'
        elif self.key_state_handler[key.DOWN]:
            self.last_action = 'up'
        elif self.key_state_handler[key.UP]:
            self.last_action = 'down'
        return self.last_action

    def on_key_press(self, symbol, modifiers):
        if symbol == key.LEFT:
            self.last_action = 'left'
        elif symbol == key.RIGHT:
            self.last_action = 'right'
        elif symbol == key.DOWN:
            self.last_action = 'up'
        elif symbol == key.UP:
            self.last_action = 'down'


class WSADKeyboardActor(Actor):
    def __init__(self, name, key_state_handler: KeyStateHandler):
        super().__init__(name)
        self.last_action = 'up'
        self.key_state_handler = key_state_handler

    def choose_action(self, game_state: list) -> str:
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


class PolicyIterationActor(Actor):
    def __init__(self, name, mdp, chkp_dir='policy'):
        super().__init__(name)

        try:
            with open(chkp_dir, 'rb') as fp:
                file_content = load(fp)
                self.V = file_content['V']
                self.policy = file_content['policy']
        except FileNotFoundError:
            self.policy, self.V = self.value_iteration(mdp)
            checkpoint = {
                'V': self.V,
                'policy': self.policy
            }
            with open(chkp_dir, 'wb') as fp:
                dump(checkpoint, fp)

    def value_iteration(self, mdp, gamma=0.9, theta=1):
        print('started')
        V = dict()
        policy = dict()
        actions = mdp.get_possible_actions()

        for state in mdp.get_all_states():
            V[str(state)] = 0
            policy[str(state)] = actions[0]
        print('initialized')
        eps = float('inf')
        last = time.perf_counter()
        while eps > theta:
            print(f'\r{eps}, last cycle took {int(last - time.perf_counter())} sec', end="")
            last = time.perf_counter()
            _V = deepcopy(V)
            for state in mdp.get_all_states():
                tempV = []
                for action in mdp.get_possible_actions(state):
                    temp_asv = 0
                    prob = 1 / len(mdp.get_next_states(state, action))
                    for _state in mdp.get_next_states(state, action):
                        temp_asv += prob * (mdp.get_reward(state, action, _state) + gamma * V[str(_state)])

                    tempV.append(temp_asv)

                V[str(state)] = max(tempV)

            _eps = []
            for state in _V:
                _eps.append(abs(_V[str(state)] - V[str(state)]))
            eps = np.max(_eps)

        for state in mdp.get_all_states():
            temp_actions = {}
            for action in mdp.get_possible_actions(state):
                temp_actions[action] = 0
                prob = 1 / len(mdp.get_next_states(state, action))
                for _state in mdp.get_next_states(state, action):
                    temp_actions[action] += prob * (mdp.get_reward(state, action, _state) + gamma * V[str(_state)])
            policy[str(state)] = max(temp_actions, key=temp_actions.get)

        return policy, V

    def choose_action(self, game_state: list) -> str:
        return self.policy[str(game_state)]


class RandomSafeActor(Actor):
    __directions = {
        'down': Point(0, 1),
        'up': Point(0, -1),
        'left': Point(-1, 0),
        'right': Point(1, 0),
    }

    def __init__(self, name, player_index=1):
        super().__init__(name)
        self.index = player_index

    def choose_action(self, game_state: list) -> str:
        myself = game_state[3][self.index]

        possible = []

        for actin, dir in self.__directions.items():
            target: Point = dir + myself[0]
            if target.out_of_bounds(game_state[0], game_state[1]) or target.serialize() in myself:
                continue

            possible.append(actin)

        if len(possible):
            return random.choice(possible)

        return random.choice(list(self.__directions.keys()))





