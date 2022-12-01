import copy

from Actor import RandomSafeActor, MCTSActor
from game import GameState
from tqdm import tqdm

SIZE_X = 5
SIZE_Y = 5
SNAKES_LENGTH = 3
SQUARE_SIZE = 40
PADDING = 3
N_TEST_GAMES = 10_000
print('Generating board')
board = GameState(SIZE_X, SIZE_Y, SNAKES_LENGTH, SQUARE_SIZE, PADDING)
print('Board generated')

mctsActor = MCTSActor('MCTS actor', GameState.get_possible_actions, copy.deepcopy(board))
randomActor = RandomSafeActor('Random actor', 1)

counters = [0, 0]

actors = [mctsActor,
          randomActor]

for _ in tqdm(range(N_TEST_GAMES), desc='Test loop'):
    board.reset()
    state = board.state()
    mctsActor.reset(state)
    while not board.isFinished():
        action = mctsActor.choose_action(state)
        counters[0] += board.move(action, 0)
        counters[1] += board.move(randomActor.choose_action(state), 1)
        _state = board.state()
        mctsActor.update(_state)
        state = _state

print(f'QActor won {N_TEST_GAMES - counters[0]}/{N_TEST_GAMES} => {((N_TEST_GAMES - counters[0]) / N_TEST_GAMES)*100:.2f}% of test games')