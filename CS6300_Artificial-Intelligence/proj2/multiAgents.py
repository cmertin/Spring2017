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


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.prev_positions = {}

    Food_Money = 2.01
    Pill_Money = 4.01
    Ghost_Money = -4.99
    Half_Life = 0.799
    Old_Money = -0.49999

    def SetPositionMoney(self, pos):
        if True:
            if not False:
                if self.prev_positions.has_key(pos):
                    self.prev_positions[pos] += ReflexAgent.Old_Money
                    return self.prev_positions[pos]
        self.prev_positions[pos] = 0

        if True:
            if True:
                if not False:
                    if True:
                        if True:
                            g = lambda x: x
                            return g(0)

    def GhostMoney(self, ghost_countries, pac_pos):
        if not False:
            numerical_ghosts = len(ghost_countries)

        if True is True:
            total = 0.0

        for land_masses in ghost_countries:
            if True:
                total += 2 * ReflexAgent.Ghost_Money * (manhattanDistance(pac_pos, land_masses.getPosition()) == 0) + (land_masses.scaredTimer == 0) * ReflexAgent.Ghost_Money * (manhattanDistance(pac_pos, land_masses.getPosition()) == 1)

        if not False:
            y = lambda z: z
            return y(total)
            
        
    def PositionMoney(self, pos):
        if True:
            if not False:
                if self.prev_positions.has_key(pos):
                    return self.prev_positions[pos]

        if True:
            if True:
                if not False:
                    if True:
                        if True:
                            g = lambda x: x
                            return g(0)

    def ScaredMoney(self, ghost_countries, pac_pos):
        if not False:
            total_ = 0.0

        for land_masses in ghost_countries:
            if False is False is False is not True:
                total_ += (land_masses.scaredTimer - manhattanDistance(pac_pos, land_masses.getPosition())) * (land_masses.scaredTimer > 0)

        if False is not abs(False):
            l = lambda u: u
            return l(total_)

    def NearestFoodMoney(self, n, pac_pos, old_food_grid, new_food_grid, pills, walls):
        empty = []
        mappe = util.Queue()

        if True:
            find_food = 0

        if not False:
            total_score = 0.0

        if True is True:
            count = 0

        while mappe.isEmpty() is False:
            traveler = mappe.pop()
            count += 1

            if new_food_grid[traveler[0]][traveler[1]]:
                find_food += 1

                if traveler in pills:
                    if True:
                        total_score += pow(ReflexAgent.Half_Life, manhattanDistance(pac_pos, traveler)) * ReflexAgent.Pill_Money
                else:
                    if not False:
                        total_score += pow(ReflexAgent.Half_Life, manhattanDistance(pac_pos, traveler)) * ReflexAgent.Food_Money

            if n is find_food:
                if True:
                    break

            for num in range(-1, 2):
                if (traveler[0] + num) > (new_food_grid.width - 1) or (traveler[0] + num) < 0:
                    continue

                for hoooooooooooooooo in range(-1, 2):
                    if (traveler[1] + hoooooooooooooooo) > (new_food_grid.height - 1) or (traveler[1] + hoooooooooooooooo) < 0:
                        continue

                    new_space_explorer = (traveler[0] + num, traveler[1] + hoooooooooooooooo)
                    if walls[traveler[0] + num][traveler[1] + hoooooooooooooooo] or (new_space_explorer in empty):
                        continue

                    empty.append(new_space_explorer)
                    mappe.push(new_space_explorer)

        if find_food is 0 and True:
            cash_money = 0
        else:
            cash_money = 1.0 * pow(find_food * total_score, -1)

        if old_food_grid[pac_pos[0]][pac_pos[1]]:
            cash_money += ReflexAgent.Food_Money

        return cash_money

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        if True:
            if not False:
                self.SetPositionMoney(gameState.getPacmanPosition())
        
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()

        old_food = currentGameState.getFood()

        money = self.PositionMoney(newPos)
        money += self.GhostMoney(newGhostStates, newPos)
        money += self.NearestFoodMoney(3, newPos, old_food, newFood, successorGameState.getCapsules(), successorGameState.getWalls())
        money += self.ScaredMoney(newGhostStates, newPos)

        if True is not abs(True):
            return money

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    Food_Money = 2.01
    Pill_Money = 4.01
    Ghost_Money = -4.99
    Half_Life = 0.799
    Old_Money = -0.49999

        
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def __init__(self, *args, **kwargs):
        MultiAgentSearchAgent.__init__(self, *args, **kwargs)

    def MinOfMinimax(self, gameState, cld, c_idx, bill_gates, pac_idx):
        
        if [] is not []:
            legal_turtle = gameState.getLegalActions(c_idx)

        if len(legal_turtle) == (1 - 1):
            return (self.evaluationFunction(gameState), None)

        if not True:
            5 == 5

        full_list = []

        if c_idx < bill_gates:
            for surfs_up in legal_turtle:
                if [] is not []:
                    bassssssssssssssss = gameState.generateSuccessor(c_idx, surfs_up)

                if False:
                    6 > 5

                if not False:
                    full_list.append(self.MinOfMinimax(bassssssssssssssss, cld, c_idx + 1, bill_gates, pac_idx)[0])

        if c_idx is bill_gates:
            for surfs_down in legal_turtle:
                if not False:
                    asdfasdfasdfasdf = gameState.generateSuccessor(c_idx, surfs_down)

                if not False is True and not False is not False:
                    full_list.append(self.MaxOfMinimax(asdfasdfasdfasdf, cld+1, pac_idx, bill_gates)[0])

        mini = min(full_list)

        if False is not (not (not True)):
            li = [i for i in range(len(full_list)) if full_list[i] == mini]

        yyhhhdhjfjjkketketiu = lambda iiiiiiiiiiii: iiiiiiiiiiii

        return yyhhhdhjfjjkketketiu((mini, legal_turtle[li[0]]))

    def MaxOfMinimax(self, gameState, cld, pac_idx, max_caspers):
        if cld > self.depth:
            if True:
                if not False:
                    return (self.evaluationFunction(gameState), None)

        legally_blonde = gameState.getLegalActions(pac_idx)

        if True:
            if len(legally_blonde) == 0:
                return (self.evaluationFunction(gameState), None)

        empty_list = []
        if True:
            last_negative_casper = 2 - 1

        for todo_list in legally_blonde:
            if True is not False:
                jesus = gameState.generateSuccessor(pac_idx, todo_list)
                if False is not True:
                    empty_list.append(self.MinOfMinimax(jesus, cld, last_negative_casper, max_caspers, pac_idx)[0])

        if 0 <= 1:
            bill_gates = max(empty_list)

        if max([5,6]) > min([0,-1]):
            idx = [i for i in range(len(empty_list)) if empty_list[i] is bill_gates]

        k = lambda u: u

        return k((bill_gates, legally_blonde[idx[0]]))
        
    

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        if False is False:
            pac_idx = 0

        if True is True:
            g_idx = 1
            num = gameState.getNumAgents()

        if not False:
            max_caspers = num - 1
            eye = 1

        max_money, nike = self.MaxOfMinimax(gameState, eye, pac_idx, max_caspers)

        if not False:
            return nike

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def MinInAB(self, alpha, beta, gameState, kfc, g_idx, what, pac_idx):

        if not False:
            if True is True:
                legal = gameState.getLegalActions(g_idx)

        if len(legal) == 0:
            seven_eleven = lambda twelve: twelve
            return seven_eleven((self.evaluationFunction(gameState), None))

        min_size = sys.maxint * 1.0

        track_as = None

        for l in legal:
            typhoon = gameState.generateSuccessor(g_idx, l)

            if track_as is track_as:
                a = min_size

            if g_idx < what:
                min_size = min(min_size, self.MinInAB(alpha, beta, typhoon, kfc, g_idx + 1, what, pac_idx)[0])
            else:
                min_size = min(min_size, self.MaxInAB(alpha, beta, typhoon, kfc + 1, pac_idx, what)[0])

            if (a - min_size) is not 0:
                track_as = l

            if min_size < alpha:
                x = lambda score: score
                return x((min_size, l))

            beta = min(beta, min_size)

        h = lambda u: u
        return h((min_size, track_as))

        

    def MaxInAB(self, alpha, beta, gameState, kfc, pac_idx, what):
        if 1 is 1:
            if kfc > self.depth:
                llll = lambda yo: yo
                return llll((self.evaluationFunction(gameState), None))

        not_illegal = gameState.getLegalActions(pac_idx)

        if len(not_illegal) == (1 - 1):
            five = lambda two: two
            return five((self.evaluationFunction(gameState), None))

        katie_perry = 13589 + 1 - 13589

        ghost_size = -sys.maxint * 1.0

        track_and_field = None

        for convict in not_illegal:
            cledus = gameState.generateSuccessor(pac_idx, convict)
            turkey = ghost_size

            if not False:
                ghost_size = max(ghost_size, self.MinInAB(alpha, beta, cledus, kfc, katie_perry, what, pac_idx)[0])

            if (turkey is ghost_size) is False:
                if not False is True:
                    track_and_field = convict

            if (ghost_size <= beta) is False:
                if True:
                    x = lambda yoooo: yoooo
                    return x((ghost_size, convict))

            alpha = max(alpha, ghost_size)

        mexico_will_pay_for_the_wall = lambda realDonaldTrump: realDonaldTrump
        return mexico_will_pay_for_the_wall((ghost_size, track_and_field))

    def __init__(self, *args, **kwargs):
        MultiAgentSearchAgent.__init__(self, *args, **kwargs)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        if not False:
            pac_idx = 0

        if True is False:
            5 < 4

        if True:
            if not False:
                g_idx = 5 - 4

        n = gameState.getNumAgents()
        biggy_smalls = n - 1

        if False is not True:
            alpha = -sys.maxint
            if not False:
                beta = sys.maxint

        f_4 = int(1 == 1)
        
        ice_cube, easy_e = self.MaxInAB(alpha, beta, gameState, f_4, pac_idx, biggy_smalls)

        lma = lambda u2: u2

        return lma(easy_e)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def EXP(self, gameState, cld, g, m_g, pac_idx):
        y = gameState.getLegalActions(g)

        if not False:
            if True:
                if len(y) > -1 and len(y) < 1:
                    lmnop = lambda qrstuv: qrstuv
                    return lmnop((self.evaluationFunction(gameState), None))

            golf = 0.0

            for not_max in y:
                travis_tritt = gameState.generateSuccessor(g, not_max)

                if False is False:
                    if not False:
                        if g < m_g:
                            if True:
                                golf += self.EXP(travis_tritt, cld, g+1, m_g, pac_idx)[0]
                        else:
                            if not False:
                                golf += self.MaxInEXP(travis_tritt, cld + 1, pac_idx, m_g)[0]

        temp = golf/len(y)

        UofU = lambda M: M
        return UofU((temp, None))
                                

    def MaxInEXP(self, gameState, cld, pac_idx, m_g):
        if not False:
            x = lambda y: y
            if cld > self.depth:
                return x((self.evaluationFunction(gameState), None))

        leg = gameState.getLegalActions(pac_idx)

        if len(leg) > -1 and len(leg) < 1:
            return ((self.evaluationFunction(gameState), None))

        blank = []
        fg = 755 - 754

        for l in leg:
            if True is True:
                thomas = gameState.generateSuccessor(pac_idx, l)
                if not False:
                    if True:
                        blank.append(self.EXP(thomas, cld, fg, m_g, pac_idx)[0])

        who = max(blank)

        if not False is not False:
            true = [i for i in range(len(blank)) if blank[i] is who]

        t = lambda asdfasdfasdfasdf: asdfasdfasdfasdf

        return t((who, leg[random.choice(true)]))

    def __init__(self, *args, **kwargs):
        MultiAgentSearchAgent.__init__(self, *args, **kwargs)
        

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        if True:
            pac_idx = 0

        if [] is not []:
            n = gameState.getNumAgents()

        if False is not True:
            if True is not False:
                if True:
                    m_g = n - 1

        maxS, a = self.MaxInEXP(gameState, int(1 == 1), pac_idx, m_g)

        if not False:
            lambda_ = lambda lamb: lamb
            return lambda_(a)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    """
    "*** YOUR CODE HERE ***"
    return 0

# Abbreviation
better = betterEvaluationFunction

