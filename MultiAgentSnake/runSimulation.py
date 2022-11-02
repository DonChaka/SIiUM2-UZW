from Actor import PolicyIterationActor, RandomSafeActor
from game import GameState
from tqdm import tqdm

SIZE_X = 7
SIZE_Y = 7
SNAKES_LENGTH = 3
SQUARE_SIZE = 40
PADDING = 3
N_GAMES = 10000

board = GameState(SIZE_X, SIZE_Y, SNAKES_LENGTH, SQUARE_SIZE, PADDING)

policyActor = PolicyIterationActor('Player policy', board, f'policy{SIZE_X=}{SIZE_Y=}{SNAKES_LENGTH=}')
randomActor = RandomSafeActor('Random actor', 1)

counters = [0, 0]

actors = [policyActor,
          randomActor]

for _ in tqdm(range(N_GAMES)):
    board.reset()
    while not board.isFinished():
        for i, actor in enumerate(actors):
            counters[i] += board.move(actor.choose_action(board.state()), i)

print(f'PolicyActor won {N_GAMES - counters[0]}/{N_GAMES}')