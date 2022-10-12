# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from itertools import product
import math
import random

from networkx import NodeNotFound

import util

import networkx as nx

from game import Agent
from game import AgentState
from multiagentTestClasses import MultiagentTreeState
from pacman import GameState
from util import manhattanDistance



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        newGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = newGameState.getPacmanPosition()

        currentFood = currentGameState.getFood()
        newGhostStates = newGameState.getGhostStates()

        if newGameState.isLose():
            return float("-inf")

        if newGameState.isWin():
            return float("inf")

        score = newGameState.getScore() - currentGameState.getScore()

        if currentGameState.hasFood(newPos[0], newPos[1]):
            score += 100

        closeGhosts = [getEuclideanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates if
                       getEuclideanDistance(newPos, ghostState.getPosition()) < 3]

        if len(closeGhosts) > 0:
            score -= 500

        if action == 'Stop':
            score -= 50

        width = currentFood.width
        height = currentFood.height

        tubeLength = 10

        if action == 'South':
            for x in range(max(0, newPos[0] - 1), min(newPos[0] + 2, width)):
                for y in range(max(0, newPos[1] - tubeLength), newPos[1]):
                    if newGameState.hasFood(x, y):
                        score += 10

        if action == 'North':
            for x in range(max(0, newPos[0] - 1), min(newPos[0] + 2, width)):
                for y in range(newPos[1], min(newPos[1] + tubeLength, height)):
                    if newGameState.hasFood(x, y):
                        score += 10

        if action == 'West':
            for x in range(max(0, newPos[0] - tubeLength), newPos[0]):
                for y in range(max(0, newPos[1] - 1), min(newPos[1] + 2, height)):
                    if newGameState.hasFood(x, y):
                        score += 10

        if action == 'East':
            for x in range(newPos[0], min(newPos[0] + tubeLength, width)):
                for y in range(max(0, newPos[1] - 1), min(newPos[1] + 2, height)):
                    if newGameState.hasFood(x, y):
                        score += 10

        return score


def getEuclideanDistance(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: MultiagentTreeState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        legalActions = gameState.getLegalActions()
        results = []

        for action in legalActions:
            _gamestate = gameState.generateSuccessor(self.index, action)
            results.append(self.minim(_gamestate, 0, self.index + 1))

        return legalActions[results.index(max(results))]

    def minim(self, gameState: MultiagentTreeState, depth: int, actorIndex: int) -> float:
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        possibleActions = gameState.getLegalActions(actorIndex)
        results = []
        n_ghosts = gameState.getNumAgents() - 1

        for action in possibleActions:
            _gameState = gameState.generateSuccessor(actorIndex, action)
            if actorIndex == n_ghosts:
                results.append(self.maxim(_gameState, depth + 1))
            else:
                results.append(self.minim(_gameState, depth, actorIndex + 1))

        return min(results)

    def maxim(self, gameState: MultiagentTreeState, depth: int) -> float:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        possibleActions = gameState.getLegalActions()
        results = []

        for action in possibleActions:
            _gameState = gameState.generateSuccessor(self.index, action)
            results.append(self.minim(_gameState, depth, self.index + 1))

        return max(results)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalActions = gameState.getLegalActions()
        results = []
        v = float('-inf')
        alpha, beta = float('-inf'), float('inf')
        for action in legalActions:
            _gamestate = gameState.generateSuccessor(self.index, action)
            temp = self.minAlpha(_gamestate, 0, self.index + 1, alpha, beta)
            results.append(temp)
            v = max(v, temp)
            if v > beta:
                return action
            alpha = max(alpha, v)

        return legalActions[results.index(max(results))]

    def minAlpha(self, gameState: MultiagentTreeState, depth: int, actorIndex: int, alpha: float, beta: float) -> float:
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        possibleActions = gameState.getLegalActions(actorIndex)
        n_ghosts = gameState.getNumAgents() - 1
        v = float('inf')

        for action in possibleActions:
            _gameState = gameState.generateSuccessor(actorIndex, action)
            if actorIndex == n_ghosts:
                v = min(v, self.maxBeta(_gameState, depth + 1, alpha, beta))
            else:
                v = min(v, self.minAlpha(_gameState, depth, actorIndex + 1, alpha, beta))

            if v < alpha:
                return v

            beta = min(beta, v)

        return v

    def maxBeta(self, gameState: MultiagentTreeState, depth: int, alpha: float, beta: float) -> float:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        possibleActions = gameState.getLegalActions()
        v = float('-inf')

        for action in possibleActions:
            _gameState = gameState.generateSuccessor(self.index, action)
            v = max(v, self.minAlpha(_gameState, depth, self.index + 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legalActions = gameState.getLegalActions()
        results = []

        for action in legalActions:
            _gamestate = gameState.generateSuccessor(self.index, action)
            results.append(self.minim(_gamestate, 0, self.index + 1))

        bestScore = max(results)
        bestIndices = [index for index in range(len(results)) if results[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalActions[chosenIndex]

    def minim(self, gameState: GameState, depth: int, actorIndex: int) -> float:
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        possibleActions = gameState.getLegalActions(actorIndex)
        results = []
        n_ghosts = gameState.getNumAgents() - 1

        for action in possibleActions:
            _gameState = gameState.generateSuccessor(actorIndex, action)
            if actorIndex == n_ghosts:
                results.append(self.maxim(_gameState, depth + 1))
            else:
                results.append(self.minim(_gameState, depth, actorIndex + 1))

        return sum(results) / len(results)

    def maxim(self, gameState: GameState, depth: int) -> float:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        possibleActions = gameState.getLegalActions()
        results = []

        for action in possibleActions:
            _gameState = gameState.generateSuccessor(self.index, action)
            results.append(self.minim(_gameState, depth, self.index + 1))

        return max(results)


def clamp(x, min_v, max_v):
    return max(min_v, min(max_v, x))


def generateGraph(gameState: GameState) -> nx.Graph:
    G = nx.Graph()
    walls = gameState.getWalls()
    width, height = walls.width, walls.height
    positions = list(product(range(1, width), range(1, height)))
    G.add_nodes_from(positions)
    for x, y in positions:
        if not walls[x][y]:
            if not walls[x+1][y]:
                G.add_edge((x, y), (x+1, y))
            if not walls[x][y+1]:
                G.add_edge((x, y), (x, y+1))

    return G




def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return 999999
    if currentGameState.isLose():
        return -999999

    G = generateGraph(currentGameState)

    score = 0

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    width, height = food.width, food.height

    ghosts = currentGameState.getGhostStates()
    scaredGhosts = [ghost for ghost in ghosts if ghost.scaredTimer]
    nScaredGhosts = len(scaredGhosts)
    try:
        closestScaredGhost = min([len(nx.shortest_path(G, ghost.configuration.pos, pos)) for ghost in scaredGhosts], default=0)
    except NodeNotFound:
        closestScaredGhost = 0

    try:
        closestActiveGhost = min([len(nx.shortest_path(G, ghost.configuration.pos, pos)) for ghost in ghosts if not ghost.scaredTimer], default=0)
    except NodeNotFound:
        closestActiveGhost = 0
    try:
        closestFood = min([len(nx.shortest_path(G, (x, y), pos)) for x, y in product(range(1, width), range(1, height)) if currentGameState.hasFood(x, y)], default=0) - 1
    except NodeNotFound:
        closestFood = 0
    n_food = currentGameState.getNumFood()

    nScaredGhostsMultiplier = 100
    foodDistanceMultiplier = -10
    closestScaredGhostMultiplier = 0
    closestActiveGhostMultiplier = 0
    nFoodMultiplier = -250

    score += nScaredGhosts * nScaredGhostsMultiplier
    score += closestFood * foodDistanceMultiplier
    score += closestScaredGhost * closestScaredGhostMultiplier
    score += closestActiveGhost * closestActiveGhostMultiplier
    score += n_food * nFoodMultiplier

    return score + currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
