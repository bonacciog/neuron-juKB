#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch API API
#
# Copyright (c) 2018-2020 Musixmatch spa
# @author Loreto Parisi at musixmatch dot com
#

import os, time
import traceback
import logging
from logging.config import dictConfig
from util.singleton import SingletonMixin

# logging file format "%Y%m%d-%H%M%S"
# INFO | DEBUG | ERROR
log_level = logging.DEBUG

class MXMLogger(SingletonMixin):

    def __init__(self):
        self.logger = self.configure_logger(name='default')

    def get_logger(self, name='default'):
        """ get default logger """
        if self.logger is None:
            self.logger = self.configure_logger(name)
        return self.logger

    def configure_logger(self, name='default'):
        """ configure a logger """

        # 'file': {
        #             'level': logging.DEBUG,
        #             'class': 'logging.handlers.RotatingFileHandler',
        #             'formatter': 'default',
        #             'filename': LOG_PATH,
        #             'maxBytes': 1024,
        #             'backupCount': 3
        #         }
        dictConfig({
            'version': 1,
            'formatters': {
                'default': {'format': '%(asctime)s - %(levelname)s - %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'}
            },
            'handlers': {
                'console': {
                    'level': log_level,
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                'default': {
                    'level': log_level,
                    #'handlers': ['console', 'file']
                    'handlers': ['console'],
                    'propagate': False
                }
            },
            'disable_existing_loggers': False
        })

        # disable logging propagation to stderr
        hn = logging.NullHandler()
        hn.setLevel( log_level )

        logger = logging.getLogger(name) 
        return logger 

    def trace(self):
        '''
            print exception
        '''
        traceback.print_exc()
