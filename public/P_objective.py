import math

import numpy as np


# def P_objective(Problem, M, Input):
#     [Output, Boundary] = P_DTLZ(Problem, M, Input)
#     if Boundary == []:
#         return Output
#     else:
#         return Output, Boundary


def fitness(Problem, M, Input):
    n = 100
    Population = Input
    FunctionValue = np.zeros((Population.shape[0], M))
    if Problem == "ZDT1":
        g = 1 + (9 * np.sum(Population[:, M - 1:], axis=1)) / (n - 1)
        FunctionValue[:, 0] = Population[:, 0]
        term2 = np.sqrt(np.divide(FunctionValue[:, 0], g))
        total = np.ones(shape=term2.shape) - term2
        FunctionValue[:, 1] = np.multiply(g[:], total[:])
    elif Problem == "ZDT2":
        g = 1 + (9 * np.sum(Population[:, M - 1:], axis=1)) / (n - 1)
        FunctionValue[:, 0] = Population[:, 0]
        term2 = np.divide(FunctionValue[:, 0], g)
        term3 = np.power(term2[:], 2)
        total = np.ones(shape=term3.shape) - term3
        FunctionValue[:, 1] = np.multiply(g[:], total[:])
    elif Problem == "ZDT3":
        g = 1 + (9 * np.sum(Population[:, M - 1:], axis=1)) / (n - 1)
        FunctionValue[:, 0] = Population[:, 0]
        term2 = np.sqrt(FunctionValue[:, 0] / g[:])
        term3 = np.multiply(np.divide(FunctionValue[:, 0], g), np.sin(10 * np.pi * (Population[:, 0])))
        total = np.ones(shape=term2.shape) - term2 - term3
        FunctionValue[:, 1] = np.multiply(g[:], total[:])
    elif Problem == "test1":
        g = 1 + (10 * Population[:, 1])
        FunctionValue[:, 0] = Population[:, 0]
        term2 = np.sqrt(FunctionValue[:, 0] / g[:])
        term3 = np.multiply(np.divide(FunctionValue[:, 0], g), np.sin(8 * np.pi * (Population[:, 0])))
        total = np.ones(shape=term2.shape) - np.power(term2,2) - term3
        FunctionValue[:, 1] = np.multiply(g[:], total[:])
    elif Problem == "test2":
        for i,item in enumerate(Population):
            if item <= 1 :
                FunctionValue[i,0] = -1 * item
                FunctionValue[i,1] = (item - 5) ** 2
            elif item > 1 and item <= 3 :
                FunctionValue[i, 0] = -2 + item
                FunctionValue[i, 1] = (item - 5) ** 2
            elif item > 3 and item <= 4:
                FunctionValue[i, 0] = 4 - item
                FunctionValue[i, 1] = (item - 5) ** 2
            elif item > 4:
                FunctionValue[i, 0] = -4 + item
                FunctionValue[i, 1] = (item - 5) ** 2
    elif Problem=="test3":
        FunctionValue[:, 0] = Population[:, 0]
        g = 11 + np.power(Population[:,1],2) - 10*np.cos(2*np.pi*Population[:,1])
        h = np.zeros(shape=g.shape).astype(float)
        i = 0
        for f1,g1 in zip(FunctionValue[:,0],g):
            if f1 <= g1:
                t = f1/g1
                h[i] = 1 -np.sqrt(t)
            i += 1
        FunctionValue[:, 1] = np.multiply(g[:],h[:])
    return FunctionValue
