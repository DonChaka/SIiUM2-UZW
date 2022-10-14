import pyglet
from pyglet.window import key, Window
from pyglet.graphics import Batch
from pyglet.shapes import Rectangle
import numpy as np

from Actor import ArrowKeyboardActor, WSADKeyboardActor
from game import Game, GameState

SIZE_X = 15
SIZE_Y = 15
SQUARE_SIZE = 40
PADDING = 3

window = Window(SIZE_X * SQUARE_SIZE, SIZE_Y * SQUARE_SIZE, 'Game', resizable=False)
batch = Batch()
keys = key.KeyStateHandler()
window.push_handlers(keys)

actors = [ArrowKeyboardActor('Player 1', keys),
          WSADKeyboardActor('Player 2', keys)]

board = GameState(SIZE_X, SIZE_Y, SQUARE_SIZE, PADDING)
board.add_player(SIZE_X // 4, 0)
board.add_player(int(SIZE_X * 3 // 4), SIZE_Y - 1)
board._generate_apple()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        window.close()
    for actor in actors:
        actor.on_key_press(symbol, modifiers)


def update(dt):
    for i, actor in enumerate(actors):
        board.move(actor.choose_action(board.get_state()), i)


@window.event
def on_draw():
    window.clear()
    board.draw(batch=batch)
    batch.draw()


pyglet.clock.schedule_interval(update, 8 / 60)

pyglet.app.run()
