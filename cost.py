import numpy as np
from math import exp

# https://www.youtube.com/watch?v=bVQUSndDllU&list=PLFt_AvWsXl0frsCrmv4fKfZ2OQIwoUuYO

# Matrix
input = [] # 2x1
w1 = [] # 3x2 # outxinput
b1 = [] # 3x1 # outx1
hidden = [] # 3x1
w2 = [] # 2x3 # outxinput
b2 = [] # 2x1 # outx1
out = [] # 2x1

def sigmoid(x):
    return 1 / (1 + exp(-x))

def calculate(x, w, b):
    x = np.matmul(w, x) # Multiply Weights
    for row in range(len(x)):
        for col in range(len(x[row])):
            x[row][col] = sigmoid(x[row][col] + b[row][col]) # Add Bias
    return x

def randomize_matrix(row, col):
    return np.random.rand(row, col)

def forward_propagation():
    hidden = calculate(input, w1, b1)
    print(hidden, "Hidden")
    out = calculate(hidden, w2, b2)
    print(out, "Output")
    return out

def initialize_matrix(i, h, o):
    global w1, b1, w2, b2
    w1 = randomize_matrix(h, i)
    b1 = randomize_matrix(h, 1)
    w2 = randomize_matrix(o, h)
    b2 = randomize_matrix(o, 1)

def get_cost():
    error = []
    totalError = 0
    for row in range(len(out)):
        error.append((out[row][0] - answer[row][0]) ** 2)
        totalError += error[row]
    return error, totalError/len(out)

# w1 = [[1.76, 0.40],
#       [0.97, 2.24]]
# b1 = [[0],
#       [0]]
# w2 = [[1.86, -0.97]]
# b2 = [[0]]


input = [[1],
         [0]]
answer = [[0],
          [1]]
initialize_matrix(len(input), 3, len(answer))
out = forward_propagation()
print(get_cost())