import copy
from turtle import back
import numpy as np
from math import exp

# https://www.youtube.com/watch?v=bVQUSndDllU&list=PLFt_AvWsXl0frsCrmv4fKfZ2OQIwoUuYO Neural Networks
# https://www.javatpoint.com/pytorch-backpropagation-process-in-deep-neural-network Backpropagation



group = np.array([0, 1, 2, 3, 4, 5])
points = group[:-1].copy()
group[0] = 9
print(group)
print(points)
quit()

# Matrix
input = [] # 2x1
w1 = [] # 3x2 # outxinput
b1 = [] # 3x1 # outx1
hidden = [] # 3x1
w2 = [] # 2x3 # outxinput
b2 = [] # 2x1 # outx1
out = [] # 2x1
answer = [] # 2x1 # out

LEARNING_RATE = 0.5
EPOCH = 5000

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
    global hidden, out
    hidden = calculate(input, w1, b1)
    #print(hidden, "Hidden")
    out = calculate(hidden, w2, b2)
    #print(out, "Output")

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

def backpropagation():
    # out
    origw2 = copy.deepcopy(w2)
    for i in range(len(hidden)):
        for j in range(len(out)):
            EtotalYFinal = -(answer[j][0] - out[j][0])
            YFinalY = out[j][0] * (1 - out[j][0])
            YW = hidden[i][0]
            EtotalW = EtotalYFinal * YFinalY * YW
            w2[j][i] = w2[j][i] - LEARNING_RATE * EtotalW
        
    # hidden layer
    for i in range(len(input)):
        for j in range(len(hidden)):
            E1H1 = (2 * (0.5 * (answer[0][0] - out[0][0])) * -1 * (out[0][0] * (1 - out[0][0]))) * origw2[0][i]
            E2H1 = (2 * (0.5 * (answer[1][0] - out[1][0])) * -1 * (out[1][0] * (1 - out[1][0]))) * origw2[1][i]
            EtotalHFinal = E1H1 + E2H1
            
            HFinalH = hidden[j][0] * (1 - hidden[j][0])
            HW = input[i][0]
            EtotalWH = EtotalHFinal * HFinalH * HW
            w1[j][i] = w1[j][i] - LEARNING_RATE * EtotalWH
        
    


# w1 = [[0.15, 0.20],
#       [0.25, 0.30]]
# b1 = [[0.35],
#       [0.35]]
# w2 = [[0.40, 0.45],
#       [0.50, 0.55]]
# b2 = [[0.60],
#       [0.60]]


input = [[1],
         [0]]
answer = [[0],
          [1]]
initialize_matrix(len(input), 2, len(answer))
# forward_propagation()
# print(get_cost())
# backpropagation()
# print(w1, "w1")
# print(b1, "b1")
# print(w2, "w2")
# print(b2, "b2")

initialize_matrix(len(input), 2, len(answer))
for i in range(EPOCH):
    forward_propagation()
    print(get_cost()[1])
    backpropagation()
print(out)