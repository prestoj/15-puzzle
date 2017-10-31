import numpy as np
from board import Board
from utils import Node, PriorityQueue
import time
np.random.seed(1)

def greedy_best_first(board, heuristic):
    """
    an implementation of the greedy best first search algorithm. it uses a heuristic function to find the quickest
    way to the destination

    :param board: (Board) the board you start at
    :param heuristic: (function) the heuristic function
    :return: (list) path to solution, (int) number of explored boards
    """

    frontier = PriorityQueue()
    node = Node(board)
    frontier.add(node, heuristic(node.data))

    explored = []
    while frontier.has_next():
        node = frontier.pop()

        if node.data.is_solved():
            return node.path(), len(explored) + 1

        for move in node.data.legal_moves():
            child = Node(node.data.forecast(move), node)
            if (not frontier.has(child)) and (child.data not in explored):
                frontier.add(child, heuristic(child.data))

        explored.append(node.data)

    return None, len(explored)


def a_star(board, heuristic):
    """
    solves the board using the A* approach accompanied by the heuristic function

    :param board: board to solve
    :param heuristic: heuristic function
    :return: path to solution, and number of explored nodes
    """

    frontier = PriorityQueue()
    node = Node(board)
    frontier.add(node, heuristic(node.data) + len(node.path()) - 1)

    explored = []

    while frontier.has_next():
        node = frontier.pop()

        # check if solved
        if node.data.is_solved():
            return node.path(), len(explored) + 1

        # add children to frontier
        for move in node.data.legal_moves():
            child = Node(node.data.forecast(move), node)
            # child must not have already been explored
            if (not frontier.has(child)) and (child.data not in explored):
                frontier.add(child, heuristic(child.data) + len(child.path()) - 1)
            # if the child is already in the frontier, it can be added only if it's better
            elif frontier.has(child):
                child_value = heuristic(child.data) + len(child.path()) - 1
                if child_value < frontier.get_value(child):
                    frontier.remove(child)
                    frontier.add(child, child_value)

        explored.append(node.data)

    return None, len(explored)


def n_wrong_heuristic(board):
    """
    counts the number of tiles incorrectly placed

    excludes the 0 tile
    """

    state = board.get_board()
    indices = np.array([np.argwhere(state == i)[0] for i in range(1, 16)])
    correct_indices = np.array([[i, j] for i in range(4) for j in range(4)])[:-1]
    n_wrong = 0
    for i,pair in enumerate(indices):
        if (pair != correct_indices[i]).any():
            n_wrong += 1

    return n_wrong




def manhattan_heuristic(board):
    """
    this sums up the manhattan distances between the board's state and the solution's state.

    excludes the 0 tile
    """
    state = board.get_board()
    indices = np.array([np.argwhere(state == i)[0] for i in range(1,16)])
    correct_indices = np.array([[i, j] for i in range(4) for j in range(4)])[:-1]

    return np.abs(indices - correct_indices).sum()

