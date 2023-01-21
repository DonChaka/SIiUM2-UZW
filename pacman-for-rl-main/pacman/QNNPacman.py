import random
from abc import ABC, abstractmethod
from .Position import Position, clamp
from typing import Dict, Callable, Optional, TypedDict, Any, Iterable
import numpy as np
from datetime import datetime
from numpy import ndarray, loadtxt, savetxt
from numpy.random import normal
from .GameState import GameState
from .Direction import Direction
from .Pacman import Pacman
import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import Sequential
import os

tf.compat.v1.disable_eager_execution()


def feats_function(self, state: GameState) -> ndarray:
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
    target: Position = me

    # # Am I even in the frame
    # if target in state.walls or 0 > target.x > state.board_size[0] or 0 > target.y > state.board_size[1]:
    #     feats.append(1)
    # else:
    #     feats.append(-1)

    # Distance to the closest ghost that could hurt us
    dists = [norm(self.__manhattanDistance(target, ghost['position'])) if not ghost['is_eatable'] else 0 for ghost
             in state.ghosts]
    if len(dists):
        feats.append((min(dists)))
    else:
        feats.append(0)

    # Distance to the closest ghost that we can hurt
    dists = [norm(self.__manhattanDistance(target, ghost['position'])) if ghost['is_eatable'] else 0 for ghost in
             state.ghosts]
    if len(dists):
        feats.append((min(dists)))
    else:
        feats.append(0)

    # Distance to the closest other pacman that could hurt us
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

    # Distance to the closest big point
    feats.append(self.min_from_list(target, state.big_points, norm))

    # distance to the closest big_big point
    feats.append(self.min_from_list(target, state.big_big_points, norm))

    # distance to the closest indestructible point
    feats.append(self.min_from_list(target, state.indestructible_points, norm))

    # distance to the closest phasing point
    feats.append(self.min_from_list(target, state.phasing_points, norm))

    return np.array(feats)


class ReplayBuffer:
    def __init__(self, mem_size, state_shape, action_shape):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def create_network(lr, n_actions, state_shape, fc1, fc2):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(fc1, input_shape=(state_shape,), activation='relu'))
    model.add(tf.keras.layers.Dense(fc2, activation='relu'))
    model.add(tf.keras.layers.Dense(n_actions, activation=None))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    return model


class TensorAgent(Pacman):

    def __init__(self, state_shape, n_actions, lr, fc1=128, fc2=128, gamma=0.99, epsilon=1, eps_dec=5e-4, eps_min=0.005,
                 mem_size=10000, batch_size=64, update_rate=100, fname='tf_q_model.h5'):
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.epsilon = epsilon
        self.update_rate = update_rate
        self.action_space = [i for i in range(n_actions)]
        self.q = create_network(lr, n_actions, state_shape, fc1, fc2)
        self.q_target = create_network(lr, n_actions, state_shape, fc1, fc2)
        self.gamma = gamma
        self.save_file = os.path.join('tmp', 'tf2', fname)
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size=mem_size, state_shape=state_shape, action_shape=n_actions)
        self.update_parameters()

    def update_parameters(self):
        self.q_target.set_weights(self.q.get_weights())

    def store_transition(self, state, action, reward, _state, terminal):
        _state = feats_function(_state)
        self.memory.store_transition(state, action, reward, _state, terminal)

    def save_model(self):
        print('----- Saving model -----')
        self.q.save_weights(self.save_file)

    def load_model(self):
        print('----- Loading model -----')
        self.q.load_weights(self.save_file)

    def choose_action(self, observation, evaluate=False):
        if np.random.random() <= self.epsilon and not evaluate:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([feats_function(observation)])
            actions = self.q.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, terminal = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q.predict(states_)
        q_next = self.q_target.predict(states_)
        q_pred = self.q.predict(states)

        max_actions = np.argmax(q_eval, axis=1)
        q_target = q_pred

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma * q_next[batch_index, max_actions.astype(int)] * terminal
        self.q.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        if not self.memory.mem_cntr % self.update_rate:
            self.update_parameters()

    def make_move(self, game_state, invalid_move=False) -> Direction:
        pass

    def give_points(self, points):
        pass

    def on_win(self, result: Dict["Pacman", int]):
        pass

    def on_death(self):
        pass
