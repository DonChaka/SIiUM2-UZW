from __future__ import annotations

import copy
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict
from typing import Callable
import numpy as np
from numpy.random import normal
from pyglet.window import key
from tqdm import tqdm

from game import Point, Snake, GameState
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
    def choose_action(self, state: list) -> str:
        pass


class ArrowKeyboardActor(Actor):
    def __init__(self, name, key_state_handler: KeyStateHandler):
        super().__init__(name)
        self.last_action = 'up'
        self.key_state_handler = key_state_handler

    def choose_action(self, state: list) -> str:
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

    def choose_action(self, state: list) -> str:
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

    def choose_action(self, state: list) -> str:
        return self.policy[str(state)]


def defValue():
    return 0


def dd():
    return defaultdict(defValue)


class QLearningActor(Actor):
    def __init__(self, name, alpha, epsilon, discount, get_legal_actions, min_eps=0.1):
        super().__init__(name)
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(dd)
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.min_eps = min_eps

    def save(self, path):
        with open(path, 'wb') as fp:
            dump(self._qvalues, fp)

    def load(self, path):
        with open(path, 'rb') as fp:
            self._qvalues = load(fp)

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value

    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return 0.0

        return max([self.get_qvalue(state, action) for action in possible_actions])

    def update(self, state, action, reward, _state):
        state = str(state)
        _state = str(_state)
        gamma = self.discount
        lr = self.alpha

        Q = (1 - lr) * self.get_qvalue(state, action) + lr * (reward + gamma * self.get_value(_state))

        self.set_qvalue(state, action, Q)
        self.epsilon -= 0.000001
        self.epsilon = max(self.epsilon, self.min_eps)

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        scores = [self.get_qvalue(state, action) for action in possible_actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return possible_actions[chosenIndex]

    def choose_action(self, state):
        possible_actions = self.get_legal_actions(state)
        state = str(state)

        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon

        if random.random() < epsilon:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

    def turn_off_learning(self):
        self.min_eps = 0
        self.epsilon = 0
        self.alpha = 0


class Node:
    __directions = {
        'down': Point(0, 1),
        'up': Point(0, -1),
        'left': Point(-1, 0),
        'right': Point(1, 0),
    }

    def __init__(self, state, parent, action, action_space, game, terminal, c=0.1, n_sims=25):
        self.state: list = state
        self.parent = parent
        self.action = action
        self.children: list[Node] = []
        self.unexpanded_children: list[Node] = []
        self.n_visited = 0
        self.actions_left = action_space() if state is None else self.safe_actions(state)
        self.action_space = action_space
        self.game: GameState = game
        self.results = {0: 0, 1: 0}
        self.terminal = terminal
        self.c = c
        self.ucb = 0
        self.n_sims = n_sims

    def expand_node(self):
        action = self.actions_left.pop()
        gameState = self.game.obj_from_state(self.state)
        _states = gameState.get_next_states(self.state, action)
        for _state in _states:
            terminal = (self.state[3][0] == _state[3][0]) or (self.state[3][1] == _state[3][1])
            self.unexpanded_children.append(
                Node(_state, self, action, self.action_space, self.game, terminal, self.c, self.n_sims))
        return self.unexpanded_children

    def val(self) -> float:
        return (self.results[1] - self.results[0]) / self.n_visited

    def safe_actions(self, state):
        ret = []
        x_size = state[0]
        y_size = state[1]
        me = Snake(state[3][0][0][0], state[3][0][0][1], body=state[3][0][1:])

        for action, direction in self.__directions.items():
            target = me.head + direction
            if not (target.out_of_bounds(x_size, y_size) or me.collides_with_point(target)):
                ret.append(action)

        return ret

    def roll_out(self) -> int:
        terminal = False
        t1, t2 = 0, 0

        self.game.set_state(self.state)
        while not terminal:
            t1 = self.game.move(np.random.choice(self.safe_actions(self.game.state())), 0)
            t2 = self.game.move(np.random.choice(self.action_space()), 1)
            terminal = t1 or t2

        return 1 - t1

    def backward(self, result) -> None:
        self.n_visited += 1.
        self.results[result] += 1.
        self.ucb = (self.val() / self.n_visited) + self.c * np.sqrt(np.log(self.n_visited) / self.n_visited)
        if self.parent:
            self.parent.backward(result)

    def best_child(self) -> Node:
        return max(self.children, key=lambda x: x.ucb)

    def expand_tree(self) -> Node:
        node = self
        while not node.terminal:
            if not len(node.unexpanded_children):
                if len(node.actions_left):
                    children = node.expand_node()
                    return np.random.choice(children)
                if node.n_visited <= len(self.children) * 1.25:
                    return node.best_child()
                node = node.best_child()
            else:
                _node = np.random.choice(node.unexpanded_children)
                node.children.append(_node)
                node.unexpanded_children.remove(_node)
                return _node
        return node

    def choose_action(self, state) -> Node:
        for _ in tqdm(range(self.n_sims), disable=True):
            self.game.set_state(state)
            node = self.expand_tree()
            result = node.roll_out()
            node.backward(result)

        return self.best_child()

    def reset(self, state):
        self.state: list = state
        self.game.set_state(state)
        self.parent = None
        self.action = None
        self.children: list[Node] = []
        self.n_visited = 0
        self.actions_left = self.action_space()
        self.results = {0: 0, 1: 0}
        self.terminal = False
        self.ucb = 0


class MCTSActor(Actor):
    def __init__(self, name, action_space, game, c=0.1, n_sims=100):
        super().__init__(name)
        self.game = copy.deepcopy(game)
        self.root = Node(None, None, None, action_space, self.game, False, c, n_sims)

    def reset(self, state) -> None:
        self.root.reset(state)

    def choose_action(self, state: list) -> str:
        chosen = self.root.choose_action(state)
        return chosen.action

    def update(self, _state) -> None:
        for child in self.root.children:
            if child.state == _state:
                self.root = child
                self.root.parent = None
                return

        for child in self.root.unexpanded_children:
            if child.state == _state:
                self.root = child
                self.root.parent = None
                return

        raise KeyError(f'State {_state} not found in children of root')


def state2feats(state, action):
    directions = {
        'down': Point(0, 1),
        'up': Point(0, -1),
        'left': Point(-1, 0),
        'right': Point(1, 0),
    }

    x_size = state[0]
    y_size = state[1]
    apple = Point(state[2][0], state[2][0])
    me = Snake(state[3][0][0][0], state[3][0][0][1], body=state[3][0][1:])
    other = Snake(state[3][1][0][0], state[3][1][0][1], body=state[3][1][1:])
    target = me.head + directions[action]

    if target.out_of_bounds(x_size, y_size) or me.collides_with_point(target) or other.collides_with_point(target):
        return np.array([float(0)] * 4)

    ret = [apple.manhattanDistance(target), other.head.manhattanDistance(target)]

    for b in other.body:
        ret.append(b.manhattanDistance(target))

    return ret


class StateValueApproxActor(Actor):
    def __init__(self, name, n_weights, features_function: Callable, get_legal_actions: Callable, alpha: float = 0.1,
                 epsilon: float = 1, discount: float = 0.9, min_eps=0.1):
        super().__init__(name)
        self.get_legal_actions = get_legal_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.min_eps = min_eps
        self.weights = normal(loc=0, scale=.5, size=n_weights)
        self.features_function = features_function

    def q(self, state, action):
        return np.dot(self.weights, self.features_function(state, action))

    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return 0.0

        return max([self.q(state, action) for action in possible_actions])

    def save(self, path):
        with open(path, 'wb') as fp:
            dump([
                self.get_legal_actions,
                self.weights,
                self.features_function,
            ], fp)

    def load(self, path):
        with open(path, 'rb') as fp:
            payload = load(fp)
            self.get_legal_actions = payload[0]
            self.weights = payload[1]
            self.features_function = payload[2]

    def update(self, state, action, reward, _state):
        gamma = self.discount
        lr = self.alpha

        delta = (reward + gamma * self.q(_state, self.get_best_action(_state))) - self.q(state, action)
        self.weights += lr * delta * self.features_function(state, action)
        j = ((reward + gamma * self.q(_state, self.get_best_action(_state))) - self.q(state, action)) ** 2

        self.epsilon -= 0.000001
        self.epsilon = max(self.epsilon, self.min_eps)
        return j

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        scores = [self.q(state, action) for action in possible_actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return possible_actions[chosenIndex]

    def choose_action(self, state):
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon

        if random.random() < epsilon:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

    def turn_off_learning(self):
        self.min_eps = 0
        self.epsilon = 0
        self.alpha = 0


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

    def choose_action(self, state: list) -> str:
        myself = state[3][self.index]

        possible = []

        for actin, direction in self.__directions.items():
            target: Point = direction + myself[0]
            if target.out_of_bounds(state[0], state[1]) or target.serialize() in myself:
                continue

            possible.append(actin)

        # if len(possible):
        #     return random.choice(possible)

        return random.choice(list(self.__directions.keys()))
