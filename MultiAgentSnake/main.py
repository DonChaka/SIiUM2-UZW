import pyglet
from pyglet.window import key, Window
from pyglet.graphics import Batch
from time import perf_counter
from Actor import ArrowKeyboardActor, WSADKeyboardActor, PolicyIterationActor, RandomSafeActor, QLearningActor
from game import GameState

SIZE_X = 6
SIZE_Y = 6
SNAKES_LENGTH = 3
SQUARE_SIZE = 80
PADDING = 3

start = perf_counter()
board = GameState(SIZE_X, SIZE_Y, SNAKES_LENGTH, SQUARE_SIZE, PADDING)
stop = perf_counter()

board.reset()
print(board.state())


print(f"Board generated in {stop - start} seconds")
print(f'Number of possibile states = {len(board.get_all_states())}')

qActor = QLearningActor('Q actor', 0.1, 1.0, 0.9, GameState.get_possible_actions)
qActor.load('QLearningSIZE_X=6SIZE_Y=6SNAKES_LENGTH=3N_TRAINING_GAMES=10000000')
qActor.turn_off_learning()

start = perf_counter()
policyActor = PolicyIterationActor('Player policy', board, f'policy{SIZE_X=}{SIZE_Y=}{SNAKES_LENGTH=}')
stop = perf_counter()

window = Window(SIZE_X * SQUARE_SIZE, SIZE_Y * SQUARE_SIZE, 'Game', resizable=False)
batch = Batch()
keys = key.KeyStateHandler()
window.push_handlers(keys)

arrowActor = ArrowKeyboardActor('Player arrow', keys)
wsadActor = WSADKeyboardActor('Player wsad', keys)
randomActor = RandomSafeActor('Random actor', 1)

print(f"Policy calculated in {stop - start} seconds")

actors = [qActor,
          randomActor]

# board.add_player(SIZE_X // 4, 0)
# board.add_player(int(SIZE_X * 3 // 4), SIZE_Y - 1)

# print(f'{board.get_state()}')
# print(f'{board.state()}')

start = False
timer = 0


@window.event
def on_key_press(symbol, modifiers):
    global start
    if symbol == key.ESCAPE:
        window.close()
    if symbol == key.ENTER:
        start = True
    # for actor in actors:
    #     actor.on_key_press(symbol, modifiers)


def update(dt):
    global timer
    if board.isFinished():
        timer += dt
        if timer >= 1:
            board.reset()
            timer = 0
        else:
            return
    if start:
        for i, actor in enumerate(actors):
            if board.move(actor.choose_action(board.state()), i):
                print(f'\rActor {actor.name} lost', end='')
                break


@window.event
def on_draw():
    window.clear()
    board.draw(batch=batch)
    batch.draw()


pyglet.clock.schedule_interval(update, 4 / 60)

pyglet.app.run()
