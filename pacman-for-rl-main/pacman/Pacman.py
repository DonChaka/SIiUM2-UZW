import random
from abc import ABC, abstractmethod
from pickle import dump, load
from typing import Dict, Callable

import numpy as np
from numpy.random import normal

from .Direction import Direction

"""
A pacman is that yellow thing with a big mouth that can eat points and ghosts!
In this game, there can be more than one pacman and they can eat each other too.
"""


class Pacman(ABC):
    """
    Make your choice!
    You can make moves completely randomly if you want, the game won't allow you to make an invalid move.
    That's what invalid_move is for - it will be true if your previous choice was invalid.
    """

    @abstractmethod
    def make_move(self, game_state, invalid_move=False) -> Direction:
        pass

    """
    The game will call this once for each pacman at each time step.
    """

    @abstractmethod
    def give_points(self, points):
        pass

    @abstractmethod
    def on_win(self, result: Dict["Pacman", int]):
        pass

    """
    Do whatever you want with this info. The game will continue until all pacmans die or all points are eaten.
    """

    @abstractmethod
    def on_death(self):
        pass


class Pacman244827(Pacman):
    def __init__(
            self,
            name,
            n_weights,
            action_space: tuple = tuple(Direction),
            features_function: Callable = None,
            alpha: float = 0.1,
            epsilon: float = 1,
            eps_dec: float = 0.0001,
            eps_min: float = 0.01,
            gamma: float = 0.9
    ):
        """
        :param name: Name of the agent, returned with str() and repr()
        :param n_weights: Number of weights, must equal to number of features returned by features_function. Ignored if features_function is None
        :param action_space: Action space that agent can choose from
        :param features_function: Function returning n_weights features. Expected signature fun(state, action) -> np.ndarray. If None than will use built in fetures_function. Default value = None
        :param alpha: Learning rate for weights learning. Default value = 0.1
        :param epsilon: Initial eposilon used in epsilon-greedy action selection method. Default value = 1
        :param eps_dec: Value that will be subtracted from epsilon in every learning step. Default value = 0.0001
        :param eps_min: Minimal value for epsilon while training. Default value = 0.01
        :param gamma: Discount rate for Bellman equation. Default value = 0.9
        """
        self.name = name
        self.action_space = action_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = gamma
        self.weights = normal(loc=0, scale=.5, size=n_weights)
        self.features_function = features_function if features_function else self.__feats_function
        self.decision_cache = {
            'state': None,
            'action': None,
            'reward': None,
            '_state': None,
        }

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __feats_function(self, state, action):
        return 0

    def q(self, state, action):
        return np.dot(self.weights, self.features_function(state, action))

    # def save(self, path):
    #     with open(path, 'wb') as fp:
    #         dump([
    #             self.get_legal_actions,
    #             self.weights,
    #             self.features_function,
    #         ], fp)
    #
    # def load(self, path):
    #     with open(path, 'rb') as fp:
    #         payload = load(fp)
    #         self.get_legal_actions = payload[0]
    #         self.weights = payload[1]
    #         self.features_function = payload[2]

    def update(self, state, action, reward, _state):
        gamma = self.gamma
        lr = self.alpha
        if None in self.decision_cache.values():
            return 0

        delta = (reward + gamma * self.q(_state, self.get_best_action(_state))) - self.q(state, action)
        self.weights += lr * delta * self.features_function(state, action)
        j = ((reward + gamma * self.q(_state, self.get_best_action(_state))) - self.q(state, action)) ** 2

        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_min)
        return j

    def get_best_action(self, state):
        possible_actions = self.action_space

        scores = [self.q(state, action) for action in possible_actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return possible_actions[chosenIndex]

    def make_move(self, state, invalid_move=False) -> Direction:
        epsilon = self.epsilon
        self.decision_cache['_state'] = state

        self.update(**self.decision_cache)

        if random.random() < epsilon:
            chosen_action = random.choice(self.action_space)
        else:
            chosen_action = self.get_best_action(state)

        self.decision_cache['action'] = chosen_action
        self.decision_cache['state'] = state

        return chosen_action

    def turn_off_learning(self):
        self.eps_min = 0
        self.epsilon = 0
        self.alpha = 0

    def give_points(self, points):
        self.decision_cache['reward'] = points

    def on_win(self, result: Dict["Pacman", int]):
        pass

    def on_death(self):
        pass


"""
I hope yours will be smarter than this one...
"""


class RandomPacman(Pacman):
    def __init__(self, print_status=False) -> None:
        self.print_status = print_status

    def give_points(self, points):
        if self.print_status:
            print(f"random pacman got {points} points")

    def on_death(self):
        if self.print_status:
            print("random pacman dead")

    def on_win(self, result: Dict["Pacman", int]):
        if self.print_status:
            print("random pacman won")

    def make_move(self, game_state, invalid_move=False) -> Direction:
        return random.choice(list(Direction))  # it will make some valid move at some point
