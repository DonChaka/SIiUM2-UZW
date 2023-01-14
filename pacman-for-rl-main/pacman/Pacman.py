import random
from abc import ABC, abstractmethod
from .Position import Position, clamp
from typing import Dict, Callable, Optional, TypedDict, Any
import numpy as np
from datetime import datetime
from numpy import ndarray, loadtxt, savetxt
from numpy.random import normal
from .GameState import GameState
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
    __DIRECTIONS = {
        Direction.UP: Position(0, -1),
        Direction.DOWN: Position(0, 1),
        Direction.LEFT: Position(-1, 0),
        Direction.RIGHT: Position(1, 0),
    }

    __def_n_features = 6

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
            alpha: float = 0.1,
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
        self.decision_cache: dict[str, Any] = {
            'state': None,
            'action': None,
            'reward': None,
            '_state': None
        }
        self.__n_wins = 0
        self.__n_loses = 0

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

        # Am I even in the frame
        if target in state.walls or 0 > target.x > state.board_size[0] or 0 > target.y > state.board_size[1]:
            feats.append(1)
        else:
            feats.append(-1)

        # Distance to closest ghost that could hurt us
        dists = [norm(self.__manhattanDistance(target, ghost['position'])) if not ghost['is_eatable'] else 0 for ghost
                 in state.ghosts]
        if len(dists):
            feats.append((min(dists)))
        else:
            feats.append(0)

        # Distance to closest ghost that we can hurt
        dists = [norm(self.__manhattanDistance(target, ghost['position'])) if ghost['is_eatable'] else 0 for ghost in
                 state.ghosts]
        if len(dists):
            feats.append((min(dists)))
        else:
            feats.append(0)

        # Distance to closest other pacman that could hurt us
        dists = [norm(self.__manhattanDistance(target, other['position'])) if not other['is_eatable'] or not other[
            'is_indestructible'] else x_size + y_size for other in state.other_pacmans]
        if len(dists):
            feats.append((min(dists)))
        else:
            feats.append(0)

        # Distance to centroid of available points
        if len(state.points):
            xs = [point.x for point in state.points]
            ys = [point.y for point in state.points]
            centroid = Position(np.average(xs), np.average(ys))
            feats.append(norm(self.__manhattanDistance(target, centroid)))
            dists = [norm(self.__manhattanDistance(target, point)) for point in state.points]
            feats.append(min(dists))
        else:
            feats.append(0)
            feats.append(0)

        return np.array(feats)

    def __q(self, state: GameState, action: Direction) -> float:
        return float(np.dot(self.weights, self.features_function(state, action)))

    def __update(self, state: GameState, action: Direction, reward: int, _state: GameState) -> None:
        gamma = self.gamma
        lr = self.alpha
        if None in self.decision_cache.values():
            return

        delta = (reward + gamma * self.__q(_state, self.get_best_action(_state))) - self.__q(state, action)
        self.weights += lr * delta * self.features_function(state, action)

        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_min)

    def get_best_action(self, state: GameState) -> Direction:
        possible_actions = self.action_space

        scores = [self.__q(state, action) for action in possible_actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return possible_actions[chosenIndex]

    def make_move(self, state: GameState, invalid_move: bool = False) -> Direction:
        epsilon = self.epsilon
        self.decision_cache['_state'] = state

        self.__update(**self.decision_cache)

        if random.random() < epsilon:
            chosen_action = random.choice(self.action_space)
        else:
            chosen_action = self.get_best_action(state)

        self.decision_cache['action'] = chosen_action
        self.decision_cache['state'] = state

        return chosen_action

    def turn_off_learning(self) -> None:
        self.eps_min = 0
        self.epsilon = 0
        self.alpha = 0

    def give_points(self, points: int) -> None:
        self.decision_cache['reward'] = points

    def on_win(self, result: Dict["Pacman", int]) -> None:
        self.__n_wins += 1

    def on_death(self) -> None:
        self.__n_loses += 1

    def get_winrate(self) -> float:
        return self.__n_wins / (self.__n_loses + self.__n_wins) * 100 if self.__n_wins + self.__n_loses else 0.0

    def reset_winrate(self) -> None:
        self.__n_wins = 0
        self.__n_loses = 0

    def save(self) -> None:
        dt_string = datetime.now().strftime("%d-%m-%Y_%H;%M;%S")
        winrate_str = str(int(self.get_winrate()))
        fname = f'{self.name}_winrate_{winrate_str}_{dt_string}.txt'
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
