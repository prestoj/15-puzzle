import numpy as np
from board import Board
from intelligence import a_star, manhattan_heuristic, n_wrong_heuristic
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import time

np.random.seed(1)
"""
setup of the network
input (256) -> fully connected (512) -> fully connected (1024) -> fully connected (512) -> output (1)
using dropout along the way to avoid overfitting
"""

model = Sequential()
model.add(Dense(units=512, input_dim=256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=512, activation='relu'))

model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='adam',
              loss='mse')


# model = load_model("models/neural_network - 512x1024x512 - 35.h5") # use this to load the model


def transform(state):
    """
    just a helper function to transform the game state into a 256 element numpy array

    this way is best for the neural net since it normally doesn't understand that 15 and 14 are totally disparate
    """
    vector_state = state.flatten()

    output = []
    for i in range(16):
        one_hot = np.zeros(16)
        one_hot[np.argwhere(vector_state == i)] = 1

        output.append(one_hot)

    return np.array(output).flatten().reshape(256)


def training_data(n_boards, n_scrambles, heuristic):
    """
    solves n_boards (scrambled n_scrambles times) and then returns each of the board states along the way to the
    solution accompanied by the remaining number of steps in that solution. this is because we want the
    neural network to map the board state onto the number of steps remaining in the solution (the heuristic function).

    :param n_boards: number of board states to solve
    :param n_scrambles: number of times to scramble the boards
    :param heuristic: the heuristic to find the solution
    :return: numpy array of states and a numpy array of their corresponding remaining steps
    """

    states = []
    values = []

    boards = []
    for _ in range(n_boards):
        this_board = Board(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]))
        this_board.scramble(n_scrambles)
        boards.append(this_board)

    for board in boards:
        solution,_ = a_star(board, heuristic)
        solution = solution[:-1]
        length = len(solution)
        for i,state in enumerate(solution):
            states.append(state.get_board())
            values.append(length - i)

    for i,state in enumerate(states):
        states[i] = transform(state)

    return np.array(states), np.array(values)


def neural_heuristic(board):
    return model.predict(transform(board.get_board()).reshape((1,256)))


def train(max_scrambles, nn_dim_string):
    complete_x = []
    complete_y = []
    for i in range(1, max_scrambles+1):
        t0 = time.time()
        x_train, y_train = training_data(200, i, neural_heuristic)
        for x in x_train:
            complete_x.append(x)
        for y in y_train:
            complete_y.append(y)
        model.fit(np.array(complete_x), np.array(complete_y), epochs=25)
        if i % 5 == 0:
            model.save("models/neural_network - {} - {}.h5".format(nn_dim_string, i))
        print(i, "iterations completed out of", max_scrambles)
        print("iteration time:", time.time() - t0)



if __name__ == "__main__":

    train(35, "512x1024x512")
