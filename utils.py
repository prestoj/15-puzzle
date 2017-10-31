import numpy as np


class Node:

    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent

    def child(self, data):
        return Node(data, self)

    def path(self):
        """
        returns the data from the first node all the way down to the current node
        """
        node = self
        path_to_head = []

        while node:
            path_to_head.append(node.data)
            node = node.parent

        return list(reversed(path_to_head))


class PriorityQueue:

    def __init__(self):
        """
        the data with the lowest corresponding values will be highest in priority
        """
        self.data = []
        self.values = []

    def pop(self):
        index = np.argmin(self.values)
        self.values.pop(index)
        element = self.data.pop(index)
        return element

    def add(self, data, value):
        self.data.append(data)
        self.values.append(value)

    def has_next(self):
        return self.data != []

    def has(self, data):
        return data in self.data

    def get_value(self, data):
        return self.values[self.data.index(data)]

    def remove(self, data):
        i = self.data.index(data)
        del self.data[i]
        del self.values[i]

    def size(self):
        return len(self.data)

