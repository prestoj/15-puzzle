import numpy as np
from utils import PriorityQueue, Node
import time
np.random.seed(1)

class Board:
    """
    this is a 15 puzzle board. the board is a 4 x 4 grid with tiles numbered 1 through 15. the last
    tile (which in this Class is represented by 0) represents a blank tile. the puzzle is as follows:

        the board starts in a random state, such as below:

                        [[11  5  1  0]
                         [12 10  6 13]
                         [ 2  4 15  3]
                         [ 9 14  8  7]].

        you must move the tiles around by pushing them around the blank tile. the goal is to
        get the board back to the default state:

                        [[ 1  2  3  4]
                         [ 5  6  7  8]
                         [ 9 10 11 12]
                         [13 14 15  0]].

    """

    def __init__(self, board=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])):
        self.board = board
        self.blank_position = np.argwhere(self.board == 0)[0]

    def __repr__(self):
        return "<Board {}>".format(self.board)

    def __eq__(self, other):
        return (self.board == other.board).all()

    def move(self, direction):
        """
        moves the blank tile in the direction chosen. it's easiest to think of the blank tile as a tile
        that swaps positions with the tile it moves into.

        as an example, below is a board that moves "up":

        [[ 1  2  3  4]      [[ 1  2  3  4]
         [ 5  6  7  8]       [ 5  6  7  8]
         [ 9 10 11 12]       [ 9 10 11  0]  <- the blank tile moves up
         [13 14 15  0]]      [13 14 15 12]] <- while pushing the 12 down
            (before)            (after)

        moving will increment n_moves by one

        :param direction: (integer) direction to move the blank tile. 1 = right, 2 = up, 3 = left, 4 = down
        """

        x, y = self.blank_position # x is the row, y is the column of the blank tile

        if direction == 1:
            # moving right
            if direction in self.legal_moves():
                self.board[x][y] = self.board[x][y + 1]
                self.board[x][y + 1] = 0
                self.blank_position[1] += 1

        elif direction == 2:
            # moving up
            if direction in self.legal_moves():
                self.board[x][y] = self.board[x - 1][y]
                self.board[x - 1][y] = 0
                self.blank_position[0] -= 1

        elif direction == 3:
            # moving left
            if direction in self.legal_moves():
                self.board[x][y] = self.board[x][y - 1]
                self.board[x][y - 1] = 0
                self.blank_position[1] -= 1

        elif direction == 4:
            # moving down
            if direction in self.legal_moves():
                self.board[x][y] = self.board[x + 1][y]
                self.board[x + 1][y] = 0
                self.blank_position[0] += 1

    def get_board(self):
        """
        :return: (np.array) the current board state
        """
        return self.board

    def print(self):
        """
        prints board to console
        """
        print(self.board)

    def reset(self):
        """
        resets the board to the default state of

        [[ 1  2  3  4]
         [ 5  6  7  8]
         [ 9 10 11 12]
         [13 14 15  0]]

        """

        self.board = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])
        self.blank_position = [3,3]

    def scramble(self, n=1000):
        """
        scrambles the board by moving in a random (legal) direction (no inverses) n times

        n: (int) number of random moves
        """
        def inverse(x):
            """
            finds the inverse action of x
            1 -> 3
            2 -> 4
            3 -> 1
            4 -> 2
            """
            return ((x + 5) % 4) + 1
        last_inverse = 0
        for _ in range(n):
            move = np.random.choice([action for action in self.legal_moves() if action != last_inverse])
            self.move(move)
            last_inverse = inverse(move)


    def is_solved(self):
        """
        :return: (boolean) True if the board is in the solved state. False otherwise.
        """

        if (self.board == np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])).all():
            return True
        return False

    def legal_moves(self):
        """
        finds and returns the legal moves. a legal move is either up, down, left, or right, and doesn't
        bump into a wall

        :return: (1d array) legal moves
        """
        moves = []
        if self.blank_position[1] != 3:
            moves.append(1)
        if self.blank_position[0] != 0:
            moves.append(2)
        if self.blank_position[1] != 0:
            moves.append(3)
        if self.blank_position[0] != 3:
            moves.append(4)

        return moves

    def solved_state(self):
        """
        :return: the solved state of the 15 puzzle
        """
        return np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])

    def forecast(self, action):
        """
        :param action: (int) direction to move
        :return: (Board) the board given the action
        """
        new_board = Board(np.copy(self.board))
        new_board.move(action)
        return new_board
