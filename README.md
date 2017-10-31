# 15-puzzle

For the blog post explaining much of this code, visit https://medium.com/@prestonbjensen/solving-the-15-puzzle-e7e60a3d9782

board.py contains the Board class, which is a simulation of the 15-puzzle using numpy.

utils.py contains a couple useful data structures (Priority Queue, and Node).

intelligence.py is where the A* and pure heuristic search algorithms are, along with the number wrong heuristic and Manhattan distance heuristic.

neural_network.py uses a neural network to serve as a heuristic function. 

The models folder contains a few different neural network architectures that you can load to use as the neural heuristic function.

