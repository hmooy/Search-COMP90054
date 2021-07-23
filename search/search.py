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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    mystack = util.Stack()
    startNode = (problem.getStartState(), '', 0, [])
    mystack.push(startNode)
    visited = set()
    while mystack :
        node = mystack.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                mystack.push(newNode)
    actions = [action[1] for action in path]

    del actions[0]
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    myqueue = util.Queue()
    startNode = (problem.getStartState(), '', 0, [])
    myqueue.push(startNode)
    visited = set()
    while myqueue:
        node = myqueue.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state,action)])
                myqueue.push(newNode)
    actions = [action[1] for action in path]


def breadthFirstSearchP(problem, capsule):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    myqueue = util.Queue()
    startNode = (problem.getStartState(), '', 0, [])
    myqueue.push(startNode)
    visited = set()
    while myqueue:
        node = myqueue.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost+succCost, path + [(state,action)])
                myqueue.push(newNode)
    paths = [action[0] for action in path]
    del paths[0]
    return paths


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    #COMP90054 Task 1, Implement your A Star search algorithm here
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pQueue = util.PriorityQueue()
    startNode = (problem.getStartState(), '', 0, [])
    pQueue.push(startNode, startNode[2]+heuristic(startNode[0],problem))
    visited = set()
    while pQueue:
        node = pQueue.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                pQueue.update(newNode,heuristic(succState,problem)+newNode[2])
    actions = [action[1] for action in path]

    del actions[0]
    return actions

def aStarSearchP2(problem, heuristic=nullHeuristic):
    #COMP90054 Task 1, Implement your A Star search algorithm here
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pQueue = util.PriorityQueue()
    startNode = (problem.getStartState(), '', 0, [])
    pQueue.push(startNode, startNode[2]+manhattanHeuristic(startNode[0],problem.goal))
    visited = set()
    while pQueue:
        node = pQueue.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                pQueue.update(newNode,manhattanHeuristic(state,problem.goal)+newNode[2])

    states = [action[0] for action in path]

    del states[0]
    return states


def manhattanHeuristic(position1, position2, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position1
    xy2 = position2
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def aStarSearchP42(problem, gr, gmin, betamin):
    #COMP90054 Task 1, Implement your A Star search algorithm here
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pQueue = util.PriorityQueue()
    startNode = (problem.getStartState(), '', 0, [])
    if manhattanHeuristic(startNode[0],gr) < manhattanHeuristic(startNode[0], gmin):
        h = betamin*manhattanHeuristic(startNode[0],gmin)
    else:
        h = manhattanHeuristic(startNode[0],problem.goal)
    pQueue.push(startNode, startNode[2]+h)
    visited = set()
    while pQueue:
        node = pQueue.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                if manhattanHeuristic(succState,gr) < manhattanHeuristic(succState, gmin):
                    h2 = betamin*manhattanHeuristic(succState, gmin)
                else:
                    h2 = manhattanHeuristic(startNode[0], problem.goal)
                pQueue.update(newNode, h2+newNode[2])

    states = [action[1] for action in path]

    del states[0]
    return states


def aStarSearchPath(problem, capsule, heuristic=nullHeuristic):
    #COMP90054 Task 1, Implement your A Star search algorithm here
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pQueue = util.PriorityQueue()
    startNode = (problem.getStartState(), '', 0, [])
    pQueue.push(startNode, startNode[2]+heuristic(startNode[0],problem))
    visited = set()
    while pQueue:
        node = pQueue.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state):
                path = path + [(state, action)]
                g = cost
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                if succState in capsule:
                    g = cost
                else :
                    g = cost+succCost
                newNode = (succState, succAction, g, path + [(state, action)])
                pQueue.update(newNode,heuristic(succState,problem)+newNode[2])
    # paths = [action[0] for action in path]
    # del paths[0]
    # return paths
    return g


def recursivebfs(problem, heuristic=nullHeuristic) :
    #COMP90054 Task 2, Implement your Recursive Best First Search algorithm here
    "*** YOUR CODE HERE ***"

    inf = float('inf')
    startNode = (problem.getStartState(), '', 0, []) # state, action, cost, path, f
    result = rbfs(problem, startNode, inf, heuristic)
    return solution(result)


from operator import itemgetter


def rbfs(problem, node, f_lim, heuristic):
    inf = float('inf')
    state, action, cost, path = node
    node_f = heuristic(state, problem)
    if problem.isGoalState(state):
        node = (state, action, cost, path+[(state, action)])
        return node, None
    successors = []
    succNodes = problem.expand(state)

    for succNode in succNodes :
        succState, succAction, succCost = succNode
        f = max(heuristic(succState, problem)+cost+succCost, node_f)
        newNode = (succState, succAction, cost + succCost, path + [(state, action)])
        successors.append((f, newNode))

    if len(successors) == 0:
        return None, inf

    while True:

        successors = sorted(successors, key=itemgetter(0))
        best_f, best_node = successors[0]
        if best_f > f_lim:
            return None, best_f
        alt_f = successors[1][0]
        result, best_f = rbfs(problem, best_node, min(f_lim, alt_f), heuristic)
        successors[0] = (best_f, best_node)

        if result != None:
            return result, None


def solution(result):

    state, action, cost, path = result[0]
    actions = [action[1] for action in path]
    del actions[0]
    return actions


# def lowest_fvalue_node(nodeList,new_f):
#     min_fval = nodeList[0][4]
#     min_fval_node_index=0
#     for n in range(1,len(nodeList)):
#         if nodeList[n][4] < min_fval :
#             min_fval_node_index = n
#             min_fval = nodeList[n][4]
#     if min_fval != new_f:
#         nodeList[min_fval_node_index][4] = new_f
#     return nodeList[min_fval_node_index]
#
#
# def second_lowest_fvalue(nodeList,lowest_f):
#     secondmin_fval = float('inf')
#     for n in range(0, len(nodeList)):
#         if lowest_f < nodeList[n][4] < secondmin_fval:
#             secondmin_fval = nodeList[n][4]
#     return secondmin_fval


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
rebfs = recursivebfs
astarp = aStarSearchPath
bfsp = breadthFirstSearchP