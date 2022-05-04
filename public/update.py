# encoding: utf-8
import numpy as np
import random
from public import NDsort, P_objective


def update_v(v_, v_min, v_max, in_, in_pbest, in_gbest):
    # Update speed

    w = 0.8
    N, D = v_.shape
    r1 = np.tile(np.random.rand(N, 1), (1, D))
    r2 = np.tile(np.random.rand(N, 1), (1, D))
    # s = np.random.normal(0, 0.01, (80,80))
    v_temp = w * v_ + r1 * (in_pbest - in_) + r2 * (in_gbest - in_)
    # v_temp = v_temp + s
    # print(v_temp.shape)
    # Speed boundary processing
    Upper = np.tile(v_max, (N, 1))
    Lower = np.tile(v_min, (N, 1))
    v_temp = np.maximum(np.minimum(Upper, v_temp), Lower)  # v不存在上下限，因此是否有必要进行限制
    return v_temp


def update_in(in_, v_, in_min, in_max):
    N, D = in_.shape
    # Update location
    in_temp = in_ + v_
    # Out-of-bounds processing
    Upper = np.tile(in_max, (N, 1))
    Lower = np.tile(in_min, (N, 1))
    in_temp = np.maximum(np.minimum(Upper, in_temp), Lower)
    return in_temp


def update_pbest(in_, fitness_, in_pbest, out_pbest):
    temp = out_pbest - fitness_
    Dominate = np.int64(np.any(temp < 0, axis=1)) - np.int64(np.any(temp > 0, axis=1))

    remained_1 = Dominate == -1
    out_pbest[remained_1] = fitness_[remained_1]
    in_pbest[remained_1] = in_[remained_1]

    remained_2 = Dominate == 0
    remained_temp_rand = np.random.rand(len(Dominate), ) < 0.5
    remained_final = remained_2 & remained_temp_rand
    out_pbest[remained_final] = fitness_[remained_final]
    in_pbest[remained_final] = in_[remained_final]
    return in_pbest, out_pbest


def apply_mutation_archive(archive_in, archive_fitness, thresh, mesh_div, problem):
    mutated_particles = []
    n = len(archive_in[0])
    for particle in archive_in:
        for j in range(2):
            m_p = np.random.random(n)
            mutated_particle = particle
            # print(n)
            # rand_ind = np.random.randint(0,n,1)
            # print(rand_ind)
            flag = False
            for i,p in enumerate(m_p):
                if p>0.5:
                    mutated_particle[i] = np.random.random(1)[0]
                    flag = True
            if not flag:
                print('yes')
            if flag:
                mutated_particles.append(mutated_particle)

    mutated_particles = np.array(mutated_particles)
    if len(mutated_particles) > 0:
        childrens_fitness = P_objective.fitness(problem, 2, mutated_particles)
        archive_in, archive_fitness = update_archive_1(mutated_particles, childrens_fitness, archive_in,
                                                       archive_fitness,
                                                       thresh, mesh_div)
    return archive_in, archive_fitness


def apply_GA_archive(archive_in, archive_fitness, thresh, mesh_div, problem):
    childrens = []
    no_condidates = archive_in.shape[0]
    parents = np.array(random.sample(range(0, no_condidates), int(1* no_condidates)))
    for i in range(0, int(len(parents) / 2)):
        if 2 * i + 1 < len(parents):
            parent1 = archive_in[parents[2 * i]]
            parent2 = archive_in[parents[2 * i + 1]]
            randpoint = np.random.randint(1, len(parent2), 1)[0]
            child1 = np.zeros(shape=(len(parent1),))
            child2 = np.zeros(shape=(len(parent2),))
            # crossover
            child1[0:randpoint] = parent1[0:randpoint]
            child1[randpoint:] = parent2[randpoint:]
            child2[0:randpoint] = parent2[0:randpoint]
            child2[randpoint:] = parent1[randpoint:]
            # mutation
            for k in range(1):
                ch1 = child1
                ch2 = child2
                flag = False
                for j in range(len(child1)):
                    m_p = np.random.random(2)
                    if m_p[0] > 0.9:
                        ch1[j] = np.random.random(1)[0]
                        flag = True
                    if m_p[1] > 0.9:
                        ch2[j] = np.random.random(1)[0]
                        flag = True
                if not flag:
                    print('yes')
                childrens.append(ch1)
                childrens.append(ch2)
    childrens = np.array(childrens)
    if len(childrens) > 0:
        childrens_fitness = P_objective.fitness(problem, 2, childrens)
        archive_in, archive_fitness = update_archive_1(childrens, childrens_fitness, archive_in,
                                                       archive_fitness,
                                                       thresh, mesh_div)
    return archive_in, archive_fitness


def update_archive_1(in_, fitness_, archive_in, archive_fitness, thresh, mesh_div):
    # First, calculate the pareto boundary of the current particle swarm,
    # and add the boundary particles to the archive archiving
    total_Pop = np.vstack((archive_in, in_))
    total_Func = np.vstack((archive_fitness, fitness_))

    FrontValue_1_index = NDsort.NDSort(total_Func, total_Pop.shape[0])[0] == 1
    FrontValue_1_index = np.reshape(FrontValue_1_index, (-1,))
    archive_in = total_Pop[FrontValue_1_index]
    archive_fitness = total_Func[FrontValue_1_index]

    if archive_in.shape[0] > thresh:
        Del_index = Delete(archive_fitness, archive_in.shape[0] - thresh, mesh_div)
        archive_in = np.delete(archive_in, Del_index, 0)
        archive_fitness = np.delete(archive_fitness, Del_index, 0)
    return archive_in, archive_fitness


def Delete(archiving_fit, K, mesh_div):
    Nop, num_obj = archiving_fit.shape

    # %% Calculate the grid location of each solution
    fmax = np.max(archiving_fit, axis=0)
    fmin = np.min(archiving_fit, axis=0)
    d = (fmax - fmin) / mesh_div
    fmin = np.tile(fmin, (Nop, 1))
    d = np.tile(d, (Nop, 1))
    Gloc = np.floor((archiving_fit - fmin) / d)
    Gloc[Gloc >= mesh_div] = mesh_div - 1
    Gloc[np.isnan(Gloc)] = 0

    # Detect the grid of each solution belongs to
    _, _, Site = np.unique(Gloc, return_index=True, return_inverse=True, axis=0)

    # Calculate the crowd degree of each grid
    CrowdG = np.histogram(Site, np.max(Site) + 1)[0]
    CrowdG_ = CrowdG.copy()

    Del_index = np.zeros(Nop, ) == 1

    while np.sum(Del_index) < K:
        maxGrid = np.where(CrowdG == max(CrowdG))[0]
        Temp = np.random.randint(0, len(maxGrid))
        Grid = maxGrid[Temp]

        InGrid = np.where(Site == Grid)[0]

        Temp = np.random.randint(0, len(InGrid))
        p = InGrid[Temp]
        Del_index[p] = True
        Site[p] = -100
        CrowdG[Grid] = CrowdG[Grid] - 1

    return np.where(Del_index == 1)[0]


def update_gbest_1(archiving_in, archiving_fit, mesh_div, particals):
    Nop, num_obj = archiving_fit.shape

    # %% Calculate the grid location of each solution
    fmax = np.max(archiving_fit, axis=0)
    fmin = np.min(archiving_fit, axis=0)
    d = (fmax - fmin) / mesh_div
    fmin = np.tile(fmin, (Nop, 1))
    d = np.tile(d, (Nop, 1))
    Gloc = np.floor((archiving_fit - fmin) / d)
    Gloc[Gloc >= mesh_div] = mesh_div - 1
    Gloc[np.isnan(Gloc)] = 0

    # Detect the grid of each solution belongs to
    _, _, Site = np.unique(Gloc, return_index=True, return_inverse=True, axis=0)

    # Calculate the crowd degree of each grid
    CrowdG = np.histogram(Site, np.max(Site) + 1)[0]

    #  Roulette-wheel 1/Fitnessselection
    TheGrid = RouletteWheelSelection(particals, CrowdG)
    ReP = np.zeros(particals, )
    for i in range(particals):
        InGrid = np.where(Site == TheGrid[i])[0]
        Temp = np.random.randint(0, len(InGrid))
        ReP[i] = InGrid[Temp]
    ReP = np.int64(ReP)
    return archiving_in[ReP], archiving_fit[ReP]


def RouletteWheelSelection(N, Fitness):
    Fitness = np.reshape(Fitness, (-1,))
    Fitness = Fitness + np.minimum(np.min(Fitness), 0)
    Fitness = np.cumsum(10 / Fitness)
    Fitness = Fitness / np.max(Fitness)
    index = np.sum(np.int64(~(np.random.rand(N, 1) < Fitness)), axis=1)

    return index
