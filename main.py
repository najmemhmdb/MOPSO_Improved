#encoding: utf-8
import numpy as np
from Mopso import *
from public import  P_objective


def main():

    particals = 80 #The number of particle swarms
    cycle_ = 100 #Number of iterations
    mesh_div = 30 #Number of grid divisions
    thresh = 200 #External archive threshold
    problem = "ZDT3"
    min_ = np.zeros(shape=(100,))
    max_ = np.ones(shape=(100,))
    # min_ = np.array([0,0])
    # max_ = np.array([1,1])


    mopso_ = Mopso(particals,max_,min_,thresh,problem,mesh_div) #Particle swarm instantiation
    _,_ = mopso_.done(cycle_) #After cycle_ round iterations, pareto boundary particles



if __name__ == "__main__":
    main()