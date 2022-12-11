import copy
import time

from Actor import RandomSafeActor, MCTSActor
from game import GameState
from tqdm import tqdm

SIZE_X = 6
SIZE_Y = 6
SNAKES_LENGTH = 3
SQUARE_SIZE = 40
PADDING = 3
N_TEST_GAMES = 200
print('Generating board')
board = GameState(SIZE_X, SIZE_Y, SNAKES_LENGTH, SQUARE_SIZE, PADDING)
print('Board generated')

mctsActor = MCTSActor('MCTS actor', GameState.get_possible_actions, copy.deepcopy(board), n_sims=1000)
randomActor = RandomSafeActor('Random actor', 1)

counters = [0, 0]

actors = [mctsActor,
          randomActor]

tq = tqdm(range(N_TEST_GAMES), desc='Test loop', disable=False)
i = 0
s = 0
for _ in tq:
    board.reset()
    state = board.state()
    mctsActor.reset(state)
    while not board.isFinished():
        start = time.time_ns()
        move = mctsActor.choose_action(state)
        stop = time.time_ns()
        s += stop - start
        i += 1
        counters[0] += board.move(move, 0)
        counters[1] += board.move(randomActor.choose_action(state), 1)
        _state = board.state()
        mctsActor.update(_state)
        state = _state

print(f'QActor won {counters[1]}/{N_TEST_GAMES} => {((counters[1]) / N_TEST_GAMES)*100:.2f}% of test games')
print(f'Average time per move: {s / i / 1_000_000:.2f}ms')