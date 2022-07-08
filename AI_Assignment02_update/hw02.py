from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

## Example Agent
class ReflexAgent(Agent):

  def Action(self, gameState):

    move_candidate = gameState.getLegalActions()

    scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
    bestScore = max(scores)
    Index = [index for index in range(len(scores)) if scores[index] == bestScore]
    get_index = random.choice(Index)

    return move_candidate[get_index]

  def reflex_agent_evaluationFunc(self, currentGameState, action):

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()



def scoreEvalFunc(currentGameState):

  return currentGameState.getScore()

class AdversialSearchAgent(Agent):

  def __init__(self, getFunc ='scoreEvalFunc', depth ='2'):
    self.index = 0
    self.evaluationFunction = util.lookup(getFunc, globals())

    self.depth = int(depth)

######################################################################################

class MinimaxAgent(AdversialSearchAgent):
  """
    [문제 01] MiniMax의 Action을 구현하시오. (20점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    ret = -123456789
    for move in gameState.getLegalActions(0):
      temp = self.min_value(gameState.generateSuccessor(0,move), 0, 1)
      if ret < temp:
        ret = temp
        x = move
    return x
  
  def max_value(self, gameState, now_depth):
    if self.depth == now_depth or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    ret = -123456789
    for move in gameState.getLegalActions(0):
      temp = self.min_value(gameState.generateSuccessor(0,move), now_depth, 1)
      ret = max(ret, temp)
    return ret

  def min_value(self, gameState, now_depth, ghost_index):
    if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)
    
    ret = 123456789; n = gameState.getNumAgents()-1
    for move in gameState.getLegalActions(ghost_index):
      if ghost_index < n:
        temp = self.min_value(gameState.generateSuccessor(ghost_index,move), now_depth, ghost_index+1)
      else:
        temp = self.max_value(gameState.generateSuccessor(ghost_index,move), now_depth+1)
      ret = min(ret, temp)
    return ret
    
    ############################################################################




class AlphaBetaAgent(AdversialSearchAgent):
  """
    [문제 02] AlphaBeta의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    ret = -123456789
    for move in gameState.getLegalActions(0):
      temp = self.min_value(gameState.generateSuccessor(0,move), 0, 1, ret, 123456789)
      if ret < temp:
        ret = temp
        x = move
    return x
  
  def max_value(self, gameState, now_depth, alpha, beta):
    if self.depth == now_depth or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    ret = -123456789
    for move in gameState.getLegalActions(0):
      temp = self.min_value(gameState.generateSuccessor(0,move), now_depth, 1, alpha, beta)
      if ret < temp:
        ret = temp
        alpha = max(alpha, ret)
      if ret > beta:
        return ret
    
    return ret

  def min_value(self, gameState, now_depth, ghost_index, alpha, beta):
    if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)
    
    ret = 123456789; n = gameState.getNumAgents()-1
    for move in gameState.getLegalActions(ghost_index):
      if ghost_index < n:
        temp = self.min_value(gameState.generateSuccessor(ghost_index,move), \
                              now_depth, ghost_index+1, alpha, beta)
      else:
        temp = self.max_value(gameState.generateSuccessor(ghost_index,move), \
                              now_depth+1, alpha, beta)
      if ret > temp:
        ret = temp
        beta = min(beta, ret)
      if ret < alpha:
        return ret
    return ret


    ############################################################################



class ExpectimaxAgent(AdversialSearchAgent):
  """
    [문제 03] Expectimax의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    ret = -123456789
    for move in gameState.getLegalActions(0):
      temp = self.min_value(gameState.generateSuccessor(0,move), 0, 1)
      if ret < temp:
        ret = temp
        x = move
    return x
  
  def max_value(self, gameState, now_depth):
    if self.depth == now_depth or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    ret = -123456789
    for move in gameState.getLegalActions(0):
      temp = self.min_value(gameState.generateSuccessor(0,move), now_depth, 1)
      ret = max(ret, temp)
    return ret

  def min_value(self, gameState, now_depth, ghost_index):
    if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)
    
    ret = 0; n = gameState.getNumAgents()-1
    possibilities = gameState.getLegalActions(ghost_index)
    for move in possibilities:
      if ghost_index < n:
        ret += self.min_value(gameState.generateSuccessor(ghost_index,move), now_depth, ghost_index+1)
      else:
        ret += self.max_value(gameState.generateSuccessor(ghost_index,move), now_depth+1)
    return ret/len(possibilities)

    ############################################################################
