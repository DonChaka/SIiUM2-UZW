import numpy as plt
from Actor import RandomSafeActor, StateValueApproxActor
from game import GameState, Point, Snake
from tqdm import tqdm


def state2feats(state, action):
    def __map(value, leftMin, leftMax, rightMin, rightMax):
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        valueScaled = float(value - leftMin) / float(leftSpan)
        return rightMin + (valueScaled * rightSpan)

    directions = {
        'down': Point(0, 1),
        'up': Point(0, -1),
        'left': Point(-1, 0),
        'right': Point(1, 0),
    }

    x_size = state[0]
    y_size = state[1]

    norm = lambda x: __map(x, 0, x_size+y_size, -1, 1)

    apple = Point(state[2][0], state[2][0])
    me = Snake(state[3][0][0][0], state[3][0][0][1], body=state[3][0][1:])
    other = Snake(state[3][1][0][0], state[3][1][0][1], body=state[3][1][1:])
    target = me.head + directions[action]
    ret = []

    if target in me.body:
        ret.append(-1000)
    else:
        ret.append(0)

    ret.append(norm(apple.manhattanDistance(target)))
    ret.append(norm(other.head.manhattanDistance(target)))

    for b in me.body:
        ret.append(norm(b.manhattanDistance(other.head)))

    return plt.array(ret)


SIZE_X = 6
SIZE_Y = 6
SNAKES_LENGTH = 3
SQUARE_SIZE = 80
PADDING = 3
N_TRAINING_GAMES = 1000000
N_TEST_GAMES = 1000

board = GameState(SIZE_X, SIZE_Y, SNAKES_LENGTH, SQUARE_SIZE, PADDING)

stateValueApproxActor = StateValueApproxActor('State value approximation agent', 4, state2feats, board.get_possible_actions)
randomActor = RandomSafeActor('Random actor', 1)

counters = [0, 0]

actors = [stateValueApproxActor,
          randomActor]

for _ in tqdm(range(N_TRAINING_GAMES), desc='Training loop'):
    state = board.reset()
    while not board.isFinished():
        action = stateValueApproxActor.choose_action(state)
        counters[0] += board.move(action, 0)
        counters[1] += board.move(randomActor.choose_action(state), 1)
        _state = board.state()
        reward = board.get_reward(state, action, _state)
        stateValueApproxActor.update(state, action, reward, _state)
        state = _state

stateValueApproxActor.turn_off_learning()
print(f'QActor won {N_TRAINING_GAMES - counters[0]}/{N_TRAINING_GAMES} => {(N_TRAINING_GAMES - counters[0]) / N_TRAINING_GAMES:.2f}% of training games')


counters = [0, 0]
for _ in tqdm(range(N_TEST_GAMES), desc='Test loop'):
    state = board.reset()
    while not board.isFinished():
        action = stateValueApproxActor.choose_action(state)
        counters[0] += board.move(action, 0)
        counters[1] += board.move(randomActor.choose_action(state), 1)
        _state = board.state()
        reward = board.get_reward(state, action, _state)
        state = _state

print(f'QActor won {N_TEST_GAMES - counters[0]}/{N_TEST_GAMES} => {(N_TEST_GAMES - counters[0]) / N_TEST_GAMES:.2f}% of test games')
