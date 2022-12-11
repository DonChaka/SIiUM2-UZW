import pyglet
from pyglet.window import key, Window
from pyglet.graphics import Batch
from Actor import RandomSafeActor, MCTSActor
from game import GameState
import copy

SIZE_X = 6
SIZE_Y = 6
SNAKES_LENGTH = 3
SQUARE_SIZE = 80
PADDING = 3
N_TEST_GAMES = 10_000
print('Generating board')
board = GameState(SIZE_X, SIZE_Y, SNAKES_LENGTH, SQUARE_SIZE, PADDING)
print('Board generated')

mctsActor = MCTSActor('MCTS actor', GameState.get_possible_actions, copy.deepcopy(board), n_sims=10000)
mctsActor.reset(board.state())
randomActor = RandomSafeActor('Random actor', 1)

counters = [0, 0]

actors = [mctsActor,
          randomActor]


window = Window(SIZE_X * SQUARE_SIZE, SIZE_Y * SQUARE_SIZE, 'Game', resizable=False)
batch = Batch()
keys = key.KeyStateHandler()
window.push_handlers(keys)

start = False
timer = 0


@window.event
def on_key_press(symbol, modifiers):
    global start
    if symbol == key.ESCAPE:
        window.close()
    if symbol == key.ENTER:
        start = True


def update(dt):
    global timer
    if board.isFinished():
        timer += dt
        if timer >= 1:
            state = board.reset()
            mctsActor.reset(state)
            timer = 0
        else:
            return
    if start:
        for i, actor in enumerate(actors):
            state = board.state()
            if board.move(actor.choose_action(state), i):
                print(f'\rActor {actor.name} lost', end='')
                break
            if not i:
                _state = board.state()
                mctsActor.update(_state)


@window.event
def on_draw():
    window.clear()
    board.draw(batch=batch)
    batch.draw()


pyglet.clock.schedule_interval(update, 4 / 60)

pyglet.app.run()
