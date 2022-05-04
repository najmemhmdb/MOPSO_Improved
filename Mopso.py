import numpy as np
from public import init,update,plot,P_objective

import time

class Mopso:
    def __init__(self,particals,max_,min_,thresh,problem,mesh_div=10):


        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        self.problem = problem
        # self.max_v = (max_-min_)*0.5  #Speed limit
        # self.min_v = (max_-min_)*(-1)*0.5 #Lower limit of speed

        self.max_v = 100 * np.ones(len(max_), )  # The upper limit of the speed, there is no upper and lower limit of the speed, so the setting is very large
        self.min_v = -100 * np.ones(len(min_), )  # Lower limit of speed

        self.plot_ = plot.Plot_pareto()

    def evaluation_fitness(self):
        # print('in evaluation')
        # print(self.in_.shape)
        self.fitness_ = P_objective.fitness(self.problem, 2, self.in_)

    def initialize(self):
        # Initialize particle position
        self.in_ = init.init_designparams(self.particals,self.min_,self.max_)
        #Initialize particle velocity
        self.v_ = init.init_v(self.particals,self.max_v,self.min_v)
        #Calculate fitness ֵ
        self.evaluation_fitness()
        #Initialize individual optimal
        self.in_p,self.fitness_p = init.init_pbest(self.in_,self.fitness_)
        #Initialize external archive
        self.archive_in,self.archive_fitness = init.init_archive(self.in_,self.fitness_)
        #Initialize the global optimal
        self.in_g,self.fitness_g = update.update_gbest_1(self.archive_in,self.archive_fitness,self.mesh_div,self.particals)
    def update_(self):
        self.v_ = update.update_v(self.v_,self.min_v,self.max_v,self.in_,self.in_p,self.in_g)
        self.in_ = update.update_in(self.in_,self.v_,self.min_,self.max_)

        self.evaluation_fitness()

        self.in_p,self.fitness_p = update.update_pbest(self.in_,self.fitness_,self.in_p,self.fitness_p)


        self.archive_in, self.archive_fitness = update.update_archive_1(self.in_, self.fitness_, self.archive_in,
                                                                      self.archive_fitness,
                                                                      self.thresh, self.mesh_div)
        #### start point of modification
        # self.archive_in, self.archive_fitness = update.apply_GA_archive(self.archive_in,
        #                                                                 self.archive_fitness,
        #                                                                 self.thresh, self.mesh_div,self.problem)
        ##### just mutation
        # self.archive_in, self.archive_fitness = update.apply_mutation_archive(self.archive_in,
        #                                                                 self.archive_fitness,
        #                                                                 self.thresh, self.mesh_div,self.problem)

        self.in_g,self.fitness_g = update.update_gbest_1(self.archive_in,self.archive_fitness,self.mesh_div,self.particals)

    def done(self,cycle_):
        self.initialize()
        # self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness,-1,self.problem)
        since = time.time()
        for i in range(cycle_):
            self.update_()
            if (i+1) % 100 == 0:
                print('First', i, 'Generation completed，time consuming: ', np.round(time.time() - since, 2), "s")
                self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness,i,self.problem)
        print(len(self.archive_in))
        return self.archive_in,self.archive_fitness

