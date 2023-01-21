import random
from abc import ABC, abstractmethod
from itertools import product

from .Position import Position, clamp
from typing import Dict, Callable, Optional, TypedDict, Any, Iterable
import numpy as np
from datetime import datetime
from numpy import ndarray, loadtxt, savetxt
from numpy.random import normal
from .GameState import GameState
from .Direction import Direction


class ExtendableList(list):
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if self.__len__() < key.start + 1:
                self.extend([0] * (key.start + 1 - self.__len__()))
            if self.__len__() < key.stop + 1:
                self.extend([0] * (key.stop - self.__len__()))
        else:
            while self.__len__() < key + 1:
                self.extend([0] * (key + 1 - self.__len__()))
        super().__setitem__(key, value)


class ReplayBuffer:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros(self.mem_size, dtype=np.object)
        self.new_state_memory = np.zeros(self.mem_size, dtype=np.object)
        self.action_memory = np.zeros(self.mem_size, dtype=np.object)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state: ndarray, action: Direction, reward: float, _state: ndarray,
                         terminal: int) -> None:
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = _state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int) -> tuple:
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


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
    __DIRECTIONS = {
        Direction.UP: Position(0, -1),
        Direction.DOWN: Position(0, 1),
        Direction.LEFT: Position(-1, 0),
        Direction.RIGHT: Position(1, 0),
    }

    __def_n_features = 17

    @staticmethod
    def __manhattanDistance(start: Position, other: Position) -> float:
        return abs(start.x - other.x) + abs(start.y - other.y)

    @staticmethod
    def __get_weights_from_file(fname: str) -> ndarray:
        return loadtxt(fname=fname, delimiter=',')

    def __init__(
            self,
            name: str = "Pacman_244827",
            n_weights: int = 1,
            action_space: tuple = tuple(Direction),
            features_function: Callable = None,
            alpha: float = 0.001,
            epsilon: float = 1,
            eps_dec: float = 0.0001,
            eps_min: float = 0.01,
            gamma: float = 0.9,
            weights_fname: Optional[str] = None
    ):
        """
        :param name: Name of the agent, returned with str() and repr()
        :param n_weights: Number of weights, must equal to number of features returned by features_function. Ignored if features_function is None, default value
        :param action_space: Action space that agent can choose from
        :param features_function: Function returning n_weights features. Expected signature fun(state, action) -> np.ndarray. If None than will use built in fetures_function. Default value = None
        :param alpha: Learning rate for weights learning. Default value = 0.1
        :param epsilon: Initial eposilon used in epsilon-greedy action selection method. Default value = 1
        :param eps_dec: Value that will be subtracted from epsilon in every learning step. Default value = 0.0001
        :param eps_min: Minimal value for epsilon while training. Default value = 0.01
        :param gamma: Discount rate for Bellman equation. Default value = 0.9
        :param weights_fname: Path to txt file containing weights. If provided, model weights will be loadewd from file, otherwise weigts will be pulled from normal distribution
        """

        self.name = name
        self.action_space = action_space
        self.legal_actions = list(action_space)

        if features_function:
            self.features_function = features_function
            self.n_features = n_weights
        else:
            self.features_function = self.__feats_function
            self.n_features = self.__def_n_features

        self.alpha = alpha
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = gamma
        if weights_fname is None:
            self.weights = normal(loc=0, scale=.5, size=self.n_features)
        else:
            self.weights = self.__get_weights_from_file(fname=weights_fname)
        self.transition_cache: dict[str, Any] = {
            'state': None,
            'action': None,
            'reward': None,
            '_state': None,
            'terminal': 0,
        }
        self.__n_wins = 0
        self.__n_loses = 0
        self.curr_score = 0
        self.scores = []
        self.idle = 0
        self._batch_size = 16
        self.memory = ReplayBuffer(100000)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __feats_function(self, state: GameState, action: Direction) -> ndarray:
        def __map(value, leftMin, leftMax, rightMin, rightMax):
            leftSpan = leftMax - leftMin
            rightSpan = rightMax - rightMin
            valueScaled = float(value - leftMin) / float(leftSpan)
            return rightMin + (valueScaled * rightSpan)

        x_size = state.board_size[0]
        y_size = state.board_size[1]

        norm = lambda x: __map(x, 0, x_size + y_size, -1, 1)

        feats = []

        me = state.you['position']
        target: Position = me + self.__DIRECTIONS[action]

        # Radius check

        radius = 3

        scared_ghosts = [ghost['position'] for ghost in state.ghosts if ghost['is_eatable']]
        angry_ghosts = [ghost['position'] for ghost in state.ghosts if not ghost['is_eatable']]
        other_pacmans = [pacman['position'] for pacman in state.other_pacmans]

        for n in range(1, radius + 1):
            n_scared_ghosts = 0
            n_angry_ghosts = 0
            n_other_pacmans = 0
            for x, y in product(range(-n, n + 1, 1), range(-n, n + 1, 1)):
                sus = Position(x, y) + target
                if 0 > sus.x > x_size or 0 > sus.y > x_size:
                    continue
                if sus in scared_ghosts:
                    n_scared_ghosts += 1

                if sus in angry_ghosts:
                    n_angry_ghosts += 1

                if sus in other_pacmans:
                    n_other_pacmans += 1

            feats.append(__map(n_scared_ghosts, 0, n * 2 + 1, -1, 1))
            feats.append(__map(n_angry_ghosts, 0, n * 2 + 1, -1, 1))
            feats.append(__map(n_other_pacmans, 0, n * 2 + 1, -1, 1))

        for n in range(1, radius + 1):
            n_points = 0
            for x, y in product(range(-n, n + 1, 1), range(-n, n + 1, 1)):
                sus = Position(x, y) + target
                if 0 > sus.x > x_size or 0 > sus.y > x_size:
                    continue
                if sus in state.points:
                    n_points += 1
            feats.append(__map(n_points, 0, n * 2 + 1, -1, 1))

        n_norm = lambda x: __map(x, 0, x_size * y_size, 0, 1)
        feats.append(n_norm(len(state.points)))


        # flags

        feats.append(1 if target in scared_ghosts else -1)
        feats.append(1 if target in angry_ghosts else -1)
        feats.append(1 if target in other_pacmans else -1)
        feats.append(1 if target in state.points else -1)

        return np.array(feats)

    def min_from_list(self, target: Position, col: Iterable, norm: Callable) -> float:
        dists = [norm(self.__manhattanDistance(target, other)) for other in col]
        if len(dists):
            return min(dists)
        return 0

    def __q(self, state: GameState, action: Direction) -> float:
        return float(np.dot(self.weights, self.features_function(state, action)))

    def __update(self, state, action, reward, _state, terminal) -> None:
        if (self.memory.mem_cntr < self._batch_size) or (None in self.transition_cache.values()) or self.alpha is 0:
            return

        gamma = self.gamma
        lr = self.alpha

        # states, actions, rewards, _states, terminals = self.memory.sample_buffer(self._batch_size)
        #
        # w_update = 0
        # delta = 0
        # for state, action, reward, _state, terminal in zip(states, actions, rewards, _states, terminals):
        #     delta = (reward + gamma * self.__q(_state, self.get_best_action(_state)) * terminal) - self.__q(state, action)
        # w_update += lr * delta * self.features_function(state, action)
        # self.weights += 1/self._batch_size * w_update

        delta = (reward + gamma * self.__q(_state, self.get_best_action(_state)) * (1 - terminal)) - self.__q(state,
                                                                                                              action)
        self.weights += lr * delta * self.features_function(state, action)

        self.epsilon -= self.eps_dec
        # self.epsilon += self.eps_dec * self.idle
        self.epsilon = min(max(self.epsilon, self.eps_min), 1)

    def get_best_action(self, state: GameState) -> Direction:
        possible_actions = self.legal_actions

        scores = [self.__q(state, action) for action in possible_actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return possible_actions[chosenIndex]

    def make_move(self, state: GameState, invalid_move: bool = False) -> Direction:
        if invalid_move:
            self.legal_actions.remove(self.transition_cache['action'])

        if not invalid_move:
            self.transition_cache['_state'] = state
        if None not in self.transition_cache.values():
            self.memory.store_transition(**self.transition_cache)
            self.__update(**self.transition_cache)

        epsilon = self.epsilon

        if random.random() < epsilon:
            chosen_action = random.choice(self.legal_actions)
        else:
            chosen_action = self.get_best_action(state)

        if not invalid_move:
            self.transition_cache['state'] = state
            self.legal_actions = list(self.action_space)

        self.transition_cache['action'] = chosen_action

        return chosen_action

    def turn_off_learning(self) -> None:
        self.epsilon = self.eps_min
        self.alpha = 0

    def give_points(self, points: int) -> None:
        if not points:
            self.idle += 1
        else:
            self.idle = max(self.idle - 5, 0)
        self.curr_score += points
        self.transition_cache['reward'] = points

    def on_game_end(self) -> None:
        self.scores.append(self.curr_score)
        self.curr_score = 0
        self.idle = 0
        self.transition_cache['terminal'] = 1
        self.memory.store_transition(**self.transition_cache)
        self.__update(**self.transition_cache)
        self.transition_cache['terminal'] = 0

    def on_win(self, result: Dict["Pacman", int]) -> None:
        self.__n_wins += 1
        self.transition_cache['reward'] = 100
        self.on_game_end()

    def on_death(self) -> None:
        self.__n_loses += 1
        self.transition_cache['reward'] = -100
        self.on_game_end()

    def get_winrate(self) -> float:
        return self.__n_wins / (self.__n_loses + self.__n_wins) * 100 if self.__n_wins + self.__n_loses else 0.0

    def get_avg_score(self):
        return np.average(self.scores)

    def reset_winrate(self) -> None:
        self.__n_wins = 0
        self.__n_loses = 0
        self.scores = []
        self.curr_score = 0

    def save(self) -> None:
        dt_string = datetime.now().strftime("%d-%m-%Y_%H;%M;%S")
        winrate_str = str(int(self.get_winrate()))
        # fname = f'{self.name}_winrate_{winrate_str}_{dt_string}.txt'
        fname = f'{self.name}.txt'
        savetxt(fname=fname, X=self.weights, delimiter=',')


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
