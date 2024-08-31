#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ File Name      :  GA.py
@ Time           :  2024/03/02 19:25:11
@ Author         :  Chunhui Zhang
@ Version        :  0.1
@ Contact        :  1772662323@qq.com
@ Description    :  description of the code
@ History        :  0.1(2024/03/02) - description of the code(your name)
'''


import numpy as np
import time
import logging
from tqdm import tqdm

from .base import SkoBase
from .tools import func_transformer
from abc import ABCMeta, abstractmethod
from .operators import crossover, mutation, ranking, selection


logging.basicConfig(filename='ga.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple(), early_stop=None, n_processes=0):
        self.func = func_transformer(func)
        assert size_pop % 2 == 0, 'size_pop must be even integer'
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim
        self.early_stop = early_stop

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(constraint_eq)  # a list of equal functions with ceq[i] = 0
        self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with c[i] <= 0

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

    @abstractmethod
    def chrom2x(self, Chrom):
        pass

    def x2y(self):
        self.Y_raw = self.func(self.X)
        mean_value = np.nanmean(self.Y_raw)
        self.Y_raw[np.isnan(self.Y_raw)] = mean_value
        print('we get self.Y_raw!!!!')
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    def run(self):
        self.max_iter = self.max_iter
        best = []
        for i in tqdm(range(self.max_iter), desc='Iteration progress'):
            t_begin = time.time()
            logging.info(f'Starting iteration {i+1}/{self.max_iter}')

            try:
                self.X = self.chrom2x(self.Chrom)
                self.Y = self.x2y()
                logging.info('self.x2y executed successfully.')
                logging.info(f'Y of iteration {i+1}:  {self.Y}')
            except Exception as e:
                logging.error(f'Error during self.x2y(data) at iteration {i+1}: {str(e)}')
                continue  # Optionally skip this iteration or use break to terminate

            try:
                self.ranking()
                self.selection()
                self.crossover()
                self.mutation()
                logging.info('Ranking, selection, crossover, mutation steps completed.')
            except Exception as e:
                logging.error(f'Error during ranking~mutation at iteration {i+1}: {str(e)}')
                continue

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            logging.info(f'best_Y of iteration {i+1}:  {self.Y[generation_best_index]}')
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        logging.info(f'Early stopping triggered after {i+1} iterations.')
                        break
                    else:
                        best.pop(0)
            logging.info(f'Time cost for iteration {i+1}: {time.time() - t_begin} seconds')


        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

    fit = run


class GA(GeneticAlgorithmBase):
    """genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint
    constraint_ueq : tuple
        unequal constraint
    precision : array_like
        The precision of every variables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    n_processes : int
        Number of processes, 0 means use all cpu
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py
    """

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None, n_processes=0):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq, early_stop, n_processes=n_processes)

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array

        # Lind is the num of genes of every variable of func（segments）
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)

        # if precision is integer:
        # if Lind_raw is integer, which means the number of all possible value is 2**n, no need to modify
        # if Lind_raw is decimal, we need ub_extend to make the number equal to 2**n,
        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        self.int_mode = np.any(self.int_mode_)
        if self.int_mode:
            self.ub_extend = np.where(self.int_mode_
                                      , self.lb + (np.exp2(self.Lind) - 1) * self.precision
                                      , self.ub)

        self.len_chrom = sum(self.Lind)

        self.crtbp()

    def crtbp(self):
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        return self.Chrom

    def gray2rv(self, gray_code):
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)

        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]

            if Chrom_temp.size == 0:
                X[:, i] = self.ub[i]
            else:
                X[:, i] = self.gray2rv(Chrom_temp)

        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
            # the ub may not obey precision, which is ok.
            # for example, if precision=2, lb=0, ub=5, then x can be 5
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

    def to(self, device):
        '''
        use pytorch to get parallel performance
        '''
        try:
            import torch
            from .operators_gpu import crossover_gpu, mutation_gpu, selection_gpu, ranking_gpu
        except:
            print('pytorch is needed')
            return self

        self.device = device
        self.Chrom = torch.tensor(self.Chrom, device=device, dtype=torch.int8)

        def chrom2x(self, Chrom):
            '''
            We do not intend to make all operators as tensor,
            because objective function is probably not for pytorch
            '''
            Chrom = Chrom.cpu().numpy()
            cumsum_len_segment = self.Lind.cumsum()
            X = np.zeros(shape=(self.size_pop, self.n_dim))
            for i, j in enumerate(cumsum_len_segment):
                if i == 0:
                    Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                else:
                    Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                X[:, i] = self.gray2rv(Chrom_temp)

            if self.int_mode:
                X = self.lb + (self.ub_extend - self.lb) * X
                X = np.where(X > self.ub, self.ub, X)
            else:
                X = self.lb + (self.ub - self.lb) * X
            return X

        self.register('mutation', mutation_gpu.mutation). \
            register('crossover', crossover_gpu.crossover_2point_bit). \
            register('chrom2x', chrom2x)

        return self
