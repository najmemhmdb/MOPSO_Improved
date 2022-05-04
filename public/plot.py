import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
class Plot_pareto:
    def __init__(self):
        self.start_time = time.time()

    def show(self,in_,fitness_,archive_in,archive_fitness,i,problem):
        if problem == "ZDT3":
            pareto = pd.read_excel('ZDT3.xlsx')
            pareto = np.array(pareto)
        elif problem == "ZDT2" :
            f = open('ZDT2.txt')
            data = f.readlines()
            pareto = np.zeros([1000,2])
            for i,d in enumerate(data):
              line = d.split(' ')
              pareto[i][0] = float(line[0])
              pareto[i][1] = float(line[1])
        elif problem == "ZDT1":
            f = open('ZDT1.txt')
            data = f.readlines()
            pareto = np.zeros([1001,2])
            for i,d in enumerate(data):
              line = d.split(' ')
              pareto[i][0] = float(line[0])
              pareto[i][1] = float(line[1])
        fig, ax = plt.subplots(ncols=2)
        pareto_min, pareto_max = min(pareto[:,1]),max(pareto[:,1])
        archive_min, archive_max = min(archive_fitness[:,1]),max(archive_fitness[:,1])
        print(archive_max-pareto_max,archive_min-pareto_min)
        ax[0].set_xlabel('F1')
        ax[0].set_ylabel('F2')
        ax[0].scatter(archive_fitness[:,0],archive_fitness[:,1],s=5, c='red', marker="*",alpha = 1.0,zorder=5)
        ax[0].plot(pareto[:, 0], pareto[:, 1], linewidth=1, color="green",zorder=20)
        ax[1].scatter(fitness_[:, 0], fitness_[:, 1], s=5, c='blue', marker="o", zorder=10)
        ax[1].plot(pareto[:, 0], pareto[:, 1], linewidth=1, color="orange", zorder=20)
        ax[1].set_xlabel('F1')
        ax[1].set_ylabel('F2')
        ax[0].set_title("REP")
        # ax[0].set_xlim(-0.05,1)
        # ax[0].set_ylim(0,2.5)
        # ax[1].set_xlim(-0.05, 1)
        # ax[1].set_ylim(0, 2.5)
        ax[1].set_title("population")
        fig.suptitle(problem)
        # ax[0].set(aspect='equal',adjustable='box')
        # ax[1].set(aspect='equal',adjustable='box')
        plt.tight_layout()

        plt.show()
        plt.ion()
