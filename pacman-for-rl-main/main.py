from pacman.Ghost import Ghosts
from pacman.Pacman import RandomPacman, Pacman244827
from pacman.Game import Game
from tqdm import tqdm


board = ["*   g",
         "gwww ",
         " w*  ",
         " www ",
         "p + p"]

board_big = ["wwwwwwwwwwwwwwwwwwwwwwwwwwww",
             "wp***********ww***********pw",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w+wwww*wwwww*ww*wwwww*wwww+w",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w**************************w",
             "w*wwww*ww*wwwwwwww*ww*wwww*w",
             "w*wwww*ww*wwwwwwww*ww*wwww*w",
             "w*****iww****ww****wwd*****w",
             "wwwwww*wwwww ww wwwww*wwwwww",
             "wwwwww*wwwww ww wwwww*wwwwww",
             "wwwwww*ww          ww*wwwwww",
             "wwwwww*ww www  www ww*wwwwww",
             "wwwwww*ww wwwggwww ww*wwwwww",
             "   z  *   www  www   *  z   ",
             "wwwwww*ww wwwggwww ww*wwwwww",
             "wwwwww*ww wwwwwwww ww*wwwwww",
             "wwwwww*ww s      s ww*wwwwww",
             "wwwwww*ww wwwwwwww ww*wwwwww",
             "wwwwww*ww wwwwwwww ww*wwwwww",
             "w*****i******ww******d*****w",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w+**ww****************ww**+w",
             "www*ww*ww*wwwwwwww*ww*ww*www",
             "www*ww*ww*wwwwwwww*ww*ww*www",
             "w******ww****ww****ww******w",
             "w*wwwwwwwwww*ww*wwwwwwwwww*w",
             "w*wwwwwwwwww*ww*wwwwwwwwww*w",
             "wp************************pw",
             "wwwwwwwwwwwwwwwwwwwwwwwwwwww"]


N_TRAIN_GAMES = 250
N_TEST_GAMES = 100

agent = Pacman244827()
try:
    for _ in tqdm(range(N_TRAIN_GAMES)):
        game = Game(board_big, [Ghosts.RED, Ghosts.PINK, Ghosts.BLUE, Ghosts.ORANGE],
                    [agent, RandomPacman(), RandomPacman(), RandomPacman()], True, delay=0)
        game.run()

    print(f'agent train winrate: {agent.get_winrate():.2f}')
    print(f'agent avg score: {agent.get_avg_score():.2f}')
    agent.reset_winrate()
    agent.turn_off_learning()
    agent.save()

    for _ in tqdm(range(N_TEST_GAMES)):
        game = Game(board_big, [Ghosts.RED, Ghosts.PINK, Ghosts.BLUE, Ghosts.ORANGE],
                    [agent, RandomPacman(), RandomPacman(), RandomPacman()], True, delay=0)
        game.run()

    print(f'agent test winrate: {agent.get_winrate():.2f}')
    print(f'agent avg score: {agent.get_avg_score():.2f}')

except KeyboardInterrupt:
    print('Program terminated')
    print(f'agent train winrate: {agent.get_winrate():.2f}')
    print(f'agent avg score: {agent.get_avg_score():.2f}')
    agent.save()






