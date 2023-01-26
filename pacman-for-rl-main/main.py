from pacman.Ghost import Ghosts
from pacman.Pacman import RandomPacman, Pacman244827, Pacman244827v1
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


N_TRAIN_GAMES = 150
N_TEST_GAMES = 100

agent0 = Pacman244827("main_pacman_0")
agent1 = Pacman244827("main_pacman_1")
agent2 = Pacman244827("main_pacman_2")
agent3 = Pacman244827("main_pacman_3")

try:
    for _ in tqdm(range(N_TRAIN_GAMES)):
        game = Game(board_big, [Ghosts.RED, Ghosts.PINK, Ghosts.BLUE, Ghosts.ORANGE],
                    [agent0, agent1, agent2, agent3], False, delay=0)
        game.run()

    print(f'agent train winrate: {agent0.get_winrate():.2f}')
    print(f'agent avg score: {agent0.get_avg_score():.2f}')
    agent0.reset_winrate()
    agent0.turn_off_learning()
    agent0.save()

    for _ in tqdm(range(N_TEST_GAMES)):
        game = Game(board_big, [Ghosts.RED, Ghosts.PINK, Ghosts.BLUE, Ghosts.ORANGE],
                    [agent0, agent1, agent2, agent3], True, delay=0)
        game.run()

    print(f'agent test winrate: {agent0.get_winrate():.2f}')
    print(f'agent avg score: {agent0.get_avg_score():.2f}')

except KeyboardInterrupt:
    print('Program terminated')
    print(f'agent train winrate: {agent0.get_winrate():.2f}')
    print(f'agent avg score: {agent0.get_avg_score():.2f}')
    agent0.save()
