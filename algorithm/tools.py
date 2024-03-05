#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ File Name      :  tools.py
@ Time           :  2024/03/02 19:25:28
@ Author         :  Chunhui Zhang
@ Version        :  0.1
@ Contact        :  1772662323@qq.com
@ Description    :  description of the code
@ History        :  0.1(2024/03/02) - description of the code(your name)
'''



import numpy as np
from functools import lru_cache
from types import MethodType, FunctionType
import warnings
import sys
import multiprocessing

if sys.platform != 'win32':
    multiprocessing.set_start_method('fork')


def set_run_mode(func, mode):
    '''

    :param func:
    :param mode: string
        can be  common, vectorization , parallel, cached
    :return:
    '''
    if mode == 'multiprocessing' and sys.platform == 'win32':
        warnings.warn('multiprocessing not support in windows, turning to multithreading')
        mode = 'multithreading'
    if mode == 'parallel':
        mode = 'multithreading'
        warnings.warn('use multithreading instead of parallel')
    func.__dict__['mode'] = mode
    return


def func_transformer(func):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    :param func:
    :return:
    '''


    mode = getattr(func, 'mode', 'others')
    valid_mode = ('common',  'multiprocessing','others')
    assert mode in valid_mode, 'valid mode should be in ' + str(valid_mode)


    if mode == 'multiprocessing':

        def func_transformed(X, train_data):
            size_pop = len(X)

            result_list = []
            result = multiprocessing.Queue()
            processes = [multiprocessing.Process(target=func, args=(train_data, X[i], result))
             for i in range(size_pop)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()
                result_list.append(result.get())


            return np.array(result_list)

        return func_transformed



    else:  # common & others
        def func_transformed(X):
            return np.array([func(x) for x in X])

        return func_transformed
