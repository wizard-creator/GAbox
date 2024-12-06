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
import warnings
import sys
import multiprocessing
# import ray
import logging
import queue

if sys.platform != 'win32':
    multiprocessing.set_start_method('fork')

logging.basicConfig(filename='tools.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



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
    valid_mode = ('common',  'multiprocessing', 'ray', 'others')
    assert mode in valid_mode, 'valid mode should be in ' + str(valid_mode)


    if mode == 'multiprocessing':

        def func_transformed(X):
            size_pop = len(X)
            result_list = []
            result = multiprocessing.Queue()
            processes = [multiprocessing.Process(target=func, args=(X[i], i, result))
                        for i in range(size_pop)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            try:
                while not result.empty():
                    output = result.get(timeout=20)
                    if output[1] is None:
                        continue
                    result_list.append(output)
            except queue.Empty:
                logging.error("Timeout occurred when getting data from the queue.")

            if len(result_list) == 0:
                logging.error("No results to process.")
                return np.array([])

            result_array = np.array(result_list)
            sorted_indices = np.argsort(result_array[:, 0])
            result_array = result_array[sorted_indices]
            print("result_array")
            return result_array[:, 1]

        return func_transformed

    # elif mode == 'ray':
    #     def func_transformed(X, train_data):
    #         size_pop = len(X)
    #         result_ids = []

    #         for i in range(size_pop):
    #             result_id = func.remote(train_data, X[i])
    #             result_ids.append(result_id)

    #         results = ray.get(result_ids)
    #         return np.array(results)

    #     return func_transformed

    elif mode == 'common':  # common & others
        def func_transformed(X):
            return func(X)

    else:
        assert False, "current method is not support!"

    return func_transformed
