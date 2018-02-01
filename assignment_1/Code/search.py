# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    from util import Stack

    start   = problem.getStartState()
    visited = [start]
    stack   = Stack()

    # initialize stack with start states' successors
    for item in problem.getSuccessors(start):
        node         = {}
        node['node'] = item
        node['path'] = [item[1]]
        stack.push(node)

    while not stack.isEmpty():
        # pop a node from the stack & expand if it hasn't been visited
        temp = stack.pop()
        node = temp['node']
        path = temp['path']

        if node[0] not in visited:
            visited.append(node[0])

            # return the path if the current node is the goal
            if problem.isGoalState(node[0]):
                return path

            # if it's not the goal, take the successors of the current node
            # and add to stack
            for successor in problem.getSuccessors(node[0]):
                to_push = {}
                temp    = path[:]
                temp.append(successor[1])
                to_push['node'] = successor
                to_push['path'] = temp
                stack.push(to_push)
    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue
    start   = problem.getStartState()
    visited = [start]
    q       = Queue()

    # initialize queue with start states' successors
    for item in problem.getSuccessors(start):
        node         = {}
        node['node'] = item
        node['path'] = [item[1]]
        q.push(node)

    while not q.isEmpty():
        # pop a node from the queue & expand if it hasn't been visited
        temp = q.pop()
        node = temp['node']
        path = temp['path']

        if node[0] not in visited:
            visited.append(node[0])

            #return the path if the current node is the goal
            if problem.isGoalState(node[0]):
                return path

            # if it's not the goal, take the successors of the current node
            # and add to queue
            for successor in problem.getSuccessors(node[0]):
                to_push = {}
                temp    = path[:]
                temp.append(successor[1])
                to_push['node'] = successor
                to_push['path'] = temp
                q.push(to_push)
    return path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue
    start   = problem.getStartState()
    visited = [start]
    q       = PriorityQueue()

    # initialize queue with start states' successors
    for item in problem.getSuccessors(start):
        node         = {}
        node['node'] = item
        node['path'] = [item[1]]
        q.push(node, node['node'][2])

    while not q.isEmpty():
        # pop a node from the queue & expand if it hasn't been visited
        temp = q.pop()
        node = temp['node']
        path = temp['path']

        if node[0] not in visited:
            visited.append(node[0])

            #return the path if the current node is the goal
            if problem.isGoalState(node[0]):
                return path

            # if it's not the goal, take the successors of the current node
            # and add to queue
            for successor in problem.getSuccessors(node[0]):
                to_push = {}
                temp    = path[:]
                temp.append(successor[1])
                to_push['node'] = successor
                to_push['path'] = temp
                q.push(to_push, to_push['node'][2])
    return path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    start   = problem.getStartState()
    visited = [start]
    q       = PriorityQueue()

    # initialize queue with start states' successors
    for item in problem.getSuccessors(start):
        node         = {}
        node['node'] = item
        node['path'] = [item[1]]
        node['dist'] = heuristic(item[0], problem)
        q.push(node, node['node'][2] + node['dist'])

    while not q.isEmpty():
        # pop a node from the queue & expand if it hasn't been visited
        temp = q.pop()
        node = temp['node']
        path = temp['path']

        if node[0] not in visited:
            visited.append(node[0])

            #return the path if the current node is the goal
            if problem.isGoalState(node[0]):
                return path

            # if it's not the goal, take the successors of the current node
            # and add to queue
            for successor in problem.getSuccessors(node[0]):
                to_push = {}
                temp    = path[:]
                temp.append(successor[1])
                to_push['node'] = successor
                to_push['path'] = temp
                to_push['dist'] = heuristic(successor[0], problem)
                q.push(to_push, to_push['node'][2] + to_push['dist'])
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
