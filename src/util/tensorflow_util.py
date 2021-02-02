#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Musixmatch AI API
#
# Copyright (c) 2020 Musixmatch spa
#

import os
import multiprocessing
import tensorflow as tf

def load_tensorflow_shared_session():
    """ Load a Tensorflow/Keras shared session """

    N_CPU = int( multiprocessing.cpu_count() / 2 )

    # OMP_NUM_THREADS controls MKL's intra-op parallelization
    # Default to available physical cores
    os.environ['OMP_NUM_THREADS'] = str( max(1, N_CPU) )

    # LP: set Tensorflow logging level
    MXM_DIST = os.getenv('MXM_DIST', 'prod')
    if MXM_DIST == 'prod':
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    else:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    # LP: create a config by gpu cpu backend
    if os.getenv('HAS_GPU', '0') == '1':
        config = tf.ConfigProto(
            device_count = { 'GPU' : 1, 'CPU': N_CPU },
            intra_op_parallelism_threads = 0,
            inter_op_parallelism_threads = N_CPU,
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
    else:
        config = tf.ConfigProto(
            device_count = { 'CPU': N_CPU, 'GPU': 0 },
            # The execution of an individual op (for some op types) can be
            # parallelized on a pool of intra_op_parallelism_threads.
            # 0 means the system picks an appropriate number.
            intra_op_parallelism_threads = N_CPU,
            # Note that the first Session created in the process sets the
            # number of threads for all future sessions unless use_per_session_threads is
            # true or session_inter_op_thread_pool is configured.
            inter_op_parallelism_threads = N_CPU,
            # use_per_session_threads will only affect the inter_op_parallelism_threads 
            # but not the intra_op_parallelism_threads. 
            use_per_session_threads = True
        )
    
    session = tf.Session(config=config)
    
    return session, config
