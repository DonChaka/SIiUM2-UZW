from Actor import RandomSafeActor, QLearningActor
from game import GameState
from tqdm import tqdm

SIZE_X = 6
SIZE_Y = 6
SNAKES_LENGTH = 3
SQUARE_SIZE = 40
PADDING = 3
N_TRAINING_GAMES = 10_000_000
N_TEST_GAMES = 1000
print('Generating board')
board = GameState(SIZE_X, SIZE_Y, SNAKES_LENGTH, SQUARE_SIZE, PADDING)
print('Board generated')

qActor = QLearningActor('Q actor', 0.1, 1.0, 0.9, GameState.get_possible_actions)
qActor.load('QLearningSIZE_X=6SIZE_Y=6SNAKES_LENGTH=3N_TRAINING_GAMES=10000000')
qActor.turn_off_learning()

randomActor = RandomSafeActor('Random actor', 1)

counters = [0, 0]

actors = [qActor,
          randomActor]

for _ in tqdm(range(N_TEST_GAMES), desc='Test loop'):
    board.reset()
    state = board.state()
    while not board.isFinished():
        action = qActor.choose_action(state)
        counters[0] += board.move(action, 0)
        counters[1] += board.move(randomActor.choose_action(state), 1)
        _state = board.state()
        reward = board.get_reward(state, action, _state)
        qActor.update(state, action, reward, _state)
        state = _state

print(f'QActor won {N_TEST_GAMES - counters[0]}/{N_TEST_GAMES} => {((N_TEST_GAMES - counters[0]) / N_TEST_GAMES)*100:.2f}% of test games')



