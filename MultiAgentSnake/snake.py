from pyglet import app
from pyglet import image
from pyglet import clock
from pyglet.window import Window, key
from random import randint

window = Window(500, 500)


@window.event
def on_draw():
    window.clear()
    draw_square(snk_x, snk_y, cell_size)
    draw_square(fd_x, fd_y, cell_size, color=(255, 0, 0, 0))


@window.event
def on_key_press(symbol, modifiers):
    global snk_dx, snk_dy

    if symbol == key.LEFT:
        snk_dx = -cell_size
        snk_dy = 0
    elif symbol == key.RIGHT:
        snk_dx = cell_size
        snk_dy = 0
    elif symbol == key.UP:
        snk_dx = 0
        snk_dy = cell_size
    elif symbol == key.DOWN:
        snk_dx = 0
        snk_dy = -cell_size

def place_food():
    global fd_x, fd_y
    fd_x = randint(0, (window.height // cell_size) - 1) * cell_size
    fd_y = randint(0, (window.width // cell_size) - 1) * cell_size

def update(dt):
    global snk_x, snk_y, snk_dx, snk_dy
    snk_x += snk_dx
    snk_y += snk_dy
    snk_dx = snk_dy = 0
    if snk_x == fd_x and snk_y == fd_y:
        place_food()


def draw_square(x, y, size, color=(0, 0, 255, 0)):
    img = image.create(size, size, image.SolidColorImagePattern(color))
    img.blit(x, y)


cell_size = 20

snk_dx, snk_dy = 0, 0

snk_x = window.width // cell_size // 2 * cell_size
snk_y = window.height // cell_size // 2 * cell_size

fd_x, fd_y = 0, 0
place_food()

clock.schedule_interval(update, 1/15)

app.run()
