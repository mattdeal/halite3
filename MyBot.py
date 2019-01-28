import hlt
from Bot16 import Bot16
from Bot18 import Bot18

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()

# pick the best bot to run for the game stats
NUM_PLAYERS = len(game.players)
if NUM_PLAYERS == 2:
    Bot16(game).run()
else:
    Bot18(game).run()
