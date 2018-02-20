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

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


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
        # this is the variable that is incremented for favourable events and then returned
        evaluation_rating = 0

        # just to initialize a big number, where other values can be compared
        import sys
        bignum = sys.maxint

        # there are few favourable events: distance to food is good, action removes food,
        # distance to ghosts is big, or distance to capsules (in the corners) is small
        # stopping could be counted as an unfavourable option

        # food is close
        shortestPathFood = bignum
        for food in newFood.asList():
            distToFood = util.manhattanDistance(food, newPos)
            if(distToFood < shortestPathFood):
                shortestPathFood = distToFood
        evaluation_rating += 10/shortestPathFood

        # moving to adjacent states removes food
        if(currentGameState.getNumFood() > successorGameState.getNumFood()):
            evaluation_rating += 150

        # non scared ghost proximity
        ghosts = currentGameState.getGhostPositions()
        shortestPathGhost = bignum
        for ghost in ghosts:
            distToGhost = util.manhattanDistance(ghost, newPos)
            if(distToGhost < shortestPathGhost):
                shortestPathGhost = distToGhost
        if(shortestPathGhost == 0):
            evaluation_rating -= (bignum - 1000)
        else:
            if shortestPathGhost < bignum:
                evaluation_rating -= 1/shortestPathGhost

        # scared ghost proximity
        # could implement a function that increments evaluation_rating if a scared ghost is near...

        # ghosts = currentGameState.getGhostPositions()
        # shortestPathGhost = bignum
        # for ghost in ghosts:
        #     distToGhost = util.manhattanDistance(ghost, newPos)
        #     if(distToGhost < shortestPathGhost):
        #         shortestPathGhost = distToGhost
        # if(shortestPathGhost == 0):
        #     evaluation_rating -= (bignum - 1000)
        # else:
        #     if shortestPathGhost < bignum:
        #         evaluation_rating -= 1/shortestPathGhost * 10

        # both ghosts are scared for more than 3 seconds so pacman is free to move around
        minimumtime = bignum
        for time in newScaredTimes:
            if time < minimumtime:
                minimumtime = time
        if minimumtime > 3:
            evaluation_rating += 500

        # distance to capsule
        successorCapsules = successorGameState.getCapsules()
        shortestPathCapsule = bignum
        for capsule in successorCapsules:
            distToCapsule = util.manhattanDistance(capsule, newPos)
            if(distToCapsule < shortestPathCapsule):
                shortestPathCapsule = distToCapsule
        evaluation_rating += 1/shortestPathCapsule * (200/150)

        # amount of capsules
        currentCapsules = currentGameState.getCapsules()
        if(len(currentCapsules) > len(successorCapsules)):
            evaluation_rating += 200


        # stopping
        if action == Directions.STOP:
            evaluation_rating -= 5

        #print evaluation_rating
        return evaluation_rating

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
      is another abstract class.numAgent = gameState.getNumAgents()
        ActionScore = []

        def _rmStop(List):
          return [x for x in List if x != 'Stop']

        def _miniMax(s, iterCount):
          if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
          if iterCount%numAgent != 0: #Ghost min
            result = 1e10
            for a in _rmStop(s.getLegalActions(iterCount%numAgent)):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = min(result, _miniMax(sdot, iterCount+1))
            return result
          else: # Pacman Max
            result = -1e10
            for a in _rmStop(s.getLegalActions(iterCount%numAgent)):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = max(result, _miniMax(sdot, iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result

        result = _miniMax(gameState, 0);
        #print _rmStop(gameState.getLegalActions(0)), ActionScore
        return _rmStop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]
        #util.raiseNotDefined()

    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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

        import sys
        bignum = sys.maxint
        agents_amount = gameState.getNumAgents()
        movement_value = []

        # Helper function for removing all occurrences of 'Stop' from a list
        def remove_stops_from_list(list):
            return [x for x in list if x != 'Stop']

        # Calculates max value for Pacman movement
        def max_val(startstate, iteration_count):
            result = bignum * (-1)
            for action in remove_stops_from_list(startstate.getLegalActions(iteration_count % agents_amount)):
                successor = startstate.generateSuccessor(iteration_count % agents_amount,action)
                result = max(result, minimax(successor, iteration_count + 1))
                if iteration_count == 0:
                    movement_value.append(result)
            return result

        # Calculates min value for ghost movement
        def min_val(startstate, iteration_count):
            result = bignum
            for action in remove_stops_from_list(startstate.getLegalActions(iteration_count % agents_amount)):
                successor = startstate.generateSuccessor(iteration_count % agents_amount, action)
                result = min(result, minimax(successor, iteration_count+1))
            return result

        # The complete minimax function that calls min_val and max_val when necessary
        # iteration_count % agents_amount tells us whether we should call minimax for pacman or for ghosts,
        # agents_amount[0] == pacman and all other agents are ghosts.
        def minimax(startstate, iteration_count):
            if iteration_count >= self.depth * agents_amount or startstate.isWin() or startstate.isLose():
                return self.evaluationFunction(startstate)
            if iteration_count % agents_amount != 0:
                #This is called for ghosts
                return min_val(startstate, iteration_count)
            else:
                #This is called for the pacman
                return max_val(startstate, iteration_count)

        minimax(gameState, 0)
        movement_direction = remove_stops_from_list(gameState.getLegalActions(0))[movement_value.index(max(movement_value))]
        #print 'movement_direction: ', movement_direction
        return movement_direction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0, -float("inf"), float("inf"))[1]

    def value(self, state, agentIndex, level, alpha, beta):
        # No need to continue processing further if game is finished
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state), "STOP")
        else:
            # If agent is pacman, use max-value
            if agentIndex == 0:
                return self.maxvalue(state, agentIndex, level, alpha, beta)
            # If agent is last ghost, use index 0 with +1 level added
            elif agentIndex == state.getNumAgents():
                return self.value(state, 0, level + 1, alpha, beta)
            # Use min-value for other cases which is ghosts
            else:
                return self.minvalue(state, agentIndex, level, alpha, beta)

    def maxvalue(self, state, agentIndex, level, alpha, beta):
        # Initalize value
        maxVal = -float("inf")
        # Get possible actions
        actions = state.getLegalActions(agentIndex)
        # No need to continue processing further if game is finished
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        # Default to stop-action
        bestAction = "Stop"
        for action in actions:
            score = self.value(state.generateSuccessor(agentIndex, action), agentIndex + 1, level, alpha, beta)
            if score[0] > maxVal:
                maxVal = score[0]
                bestAction = action
            else:
                pass
            if maxVal > beta:
               return (maxVal, bestAction)
            else:
                pass
            # Max's best option on path to root
            alpha = max(maxVal, alpha)
            result = (maxVal, bestAction)

        return result

    def minvalue(self, state, agentIndex, level, alpha, beta):
        # Initialize value
        minVal = float("inf")
        # Get possible actions
        actions = state.getLegalActions(agentIndex)

        bestAction = "Stop"
        # No need to continue processing further if game is finished
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if level == self.depth - 1 and agentIndex == (state.getNumAgents() - 1):
            # Loop for all successors of state
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                evaluation = self.evaluationFunction(successor)
                scoreAction = (evaluation, action, alpha, beta)
                if scoreAction[0] < minVal:
                    minVal = scoreAction[0]
                    bestAction = action
                    if minVal < alpha:
                        return (minVal, bestAction)
                    # Min's best option on path to root
                    beta = min(minVal, beta)
            result = (minVal, bestAction)
            return result
        else:
            # Loop for all successors of state
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                scoreAction = self.value(successor, agentIndex + 1, level, alpha, beta)
                if scoreAction[0] < minVal:
                    minVal = scoreAction[0]
                    bestAction = action
                    if minVal < alpha:
                        return (minVal, bestAction)
                    # Min's best option on path to root
                    beta = min(minVal, beta)
            result = (minVal, bestAction)
            return result

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        import sys
        bignum = sys.maxint
        agents_amount = gameState.getNumAgents()
        movement_value = []

        # Helper function for removing all occurrences of 'Stop' from a list
        def remove_stops_from_list(list):
            return [x for x in list if x != 'Stop']

        # Calculates max value for Pacman movement
        def max_val(state, iteration_count):
            result = bignum * (-1)
            for action in remove_stops_from_list(state.getLegalActions(iteration_count % agents_amount)):
                successor = state.generateSuccessor(iteration_count % agents_amount,action)
                result = max(result, expectimax(successor, iteration_count + 1))
                if iteration_count == 0:
                    movement_value.append(result)
            return result

        # Calculates exp value for ghost movement
        def exp_val(state, iteration_count):
            result = 0
            for action in remove_stops_from_list(state.getLegalActions(iteration_count % agents_amount)):
                successor = state.generateSuccessor(iteration_count % agents_amount, action)
                # probability of an action, ghosts random
                p = 1.0 / float(len(state.getLegalActions(iteration_count % agents_amount)))
                result += p * expectimax(successor, iteration_count + 1)
            return result

        def expectimax(state, iteration_count):
            if (iteration_count >= self.depth * agents_amount) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if iteration_count % agents_amount: #ghost
                return exp_val(state, iteration_count)
            else: #pacman
                return max_val(state, iteration_count)

        expectimax(gameState, 0)
        movement_direction = remove_stops_from_list(gameState.getLegalActions(0))[movement_value.index(max(movement_value))]
        #print 'movement_direction: ', movement_direction
        return movement_direction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def food_distance(gameState):
      food_distances = []
      pacman_location = gameState.getPacmanPosition()
      for food in gameState.getFood().asList():
          food_distances.append(1.0/manhattanDistance(pacman_location, food))
      if len(food_distances)>0:
          return max(food_distances)
      else:
          return 0

     ## COULD IMPLEMENT A CAPSULE DISTANCE FUNC HERE TOO ##
    # def capsule_distance(gameState):
    #     capsule_distances = []
    #     pacman_location = gameState.getPacmanPosition()
    #     for capsule in gameState.getCapsules().asList():
    #         capsule_distances.append(1.0/manhattanDistance(pacman_location,capsule))
    #     if len(capsule_distances)>0:
    #         return min(capsule_distances)
    #     else:
    #         return 0

    def collectables_amount(gameState):
        evaluation_rating = 0
        food_amount = currentGameState.getNumFood()
        capsule_amount = len(currentGameState.getCapsules())
        if food_amount != 0:
            evaluation_rating = 100.0 / food_amount
            return evaluation_rating - capsule_amount * 2
        else:
            return capsule_amount * (-2)

    def capsule_distance(gameState):
      evaluation_rating = []
      pacman_location = gameState.getPacmanPosition()
      for capsule_location in gameState.getCapsules():
          evaluation_rating.append(50.0/manhattanDistance(pacman_location, capsule_location))
      if len(evaluation_rating) > 0:
          return max(evaluation_rating)
      else:
          return 0

    def ghost_distance(gameState):
      evaluation_rating = 0
      pacman_location = gameState.getPacmanPosition()
      for ghost in gameState.getGhostStates():
          ghost_distance = manhattanDistance(pacman_location, ghost.getPosition())
          if ghost.scaredTimer == 0:
              evaluation_rating -= max(7 - ghost_distance, 0) ** 2
          else:
              evaluation_rating += max(8 - ghost_distance, 0) ** 2
      return evaluation_rating

    evaluation_rating = currentGameState.getScore() + ghost_distance(currentGameState) + food_distance(currentGameState) + capsule_distance(currentGameState)
    evaluation_rating += collectables_amount(currentGameState)
    #evaluation_rating += capsule_distance(currentGameState)
    return evaluation_rating

# Abbreviation
better = betterEvaluationFunction
