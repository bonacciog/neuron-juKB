#!/usr/bin/env python
# coding: utf8

"""
    Module that provides a class wrapper for source separation.

    :Example:

    >>> from spleeter.separator import Separator
    >>> separator = Separator('spleeter:2stems')
    >>> separator.separate(waveform, lambda instrument, data: ...)
    >>> separator.separate_to_file(...)
"""

import os, sys
import json

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from os.path import basename, join, splitext

import tensorflow as tf


#from . import SpleeterError
from spleeter_audio.adapter import get_default_audio_adapter
from spleeter_audio.convertor import to_stereo
from spleeter_model_v2 import model_fn
from spleeter_utils_v2.configuration import load_configuration
from spleeter_utils_v2.estimator import create_estimator, to_predictor
import splicer_utils as splicerutils

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

class Separator(object):
    """ A wrapper class for performing separation. """

    def __init__(self, params_descriptor, MWF=False, MODEL_DIR=None, session=None, config=None, multiprocess=False):
        """ Default constructor.

        :param params_descriptor: Descriptor for TF params to be used.
        :param MWF: (Optional) True if MWF should be used, False otherwise.
        """
        self._params = load_configuration(params_descriptor)
        if MODEL_DIR:
            self._params['model_dir'] = MODEL_DIR
        self.tf_session = session
        self.tf_config = config
        self._sample_rate = self._params['sample_rate']
        self._MWF = MWF
        self._predictor = None
        self._pool = Pool() if multiprocess else None
        self._tasks = []

    def _get_predictor(self):
        """ Lazy loading access method for internal predictor instance.

        :returns: Predictor to use for source separation.
        """
        if self._predictor is None:
            estimator = create_estimator(self._params, self._MWF, self.tf_config)
            self._predictor = to_predictor(estimator)
        return self._predictor

    def join(self, timeout=200):
        """ Wait for all pending tasks to be finished.

        :param timeout: (Optional) task waiting timeout.
        """
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)

    def separate(self, waveform):
        """ Performs source separation over the given waveform.

        The separation is performed synchronously but the result
        processing is done asynchronously, allowing for instance
        to export audio in parallel (through multiprocessing).

        Given result is passed by to the given consumer, which will
        be waited for task finishing if synchronous flag is True.

        :param waveform: Waveform to apply separation on.
        :returns: Separated waveforms.
        """
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)
        predictor = self._get_predictor()
        
        # splice waveform in 30seconds 
        #splices = splicerutils.splicer(waveform, self._sample_rate)
        splices = None

        # if waveform was spliced, execute prediction one at a time
        if isinstance(splices, dict):

            pred_splices = {}
            for idx, spliced_wave in splices.items():

                prediction = predictor({
                    'waveform': spliced_wave,
                    'audio_id': ''})
                prediction.pop('audio_id')
                pred_splices[idx] = prediction['vocals']

            # reconstruct
            reconstructed_pred = splicerutils.reconstruct_waveform(pred_splices)

            output = {}
            output['instruments'] = []
            output['vocals'] = reconstructed_pred

            # check if duration of reconstructed file matches the original one
            original_duration = splicerutils.get_duration(waveform[:, 0], self._sample_rate)
            new_duration = splicerutils.get_duration(reconstructed_pred[:, 0], self._sample_rate)

            if new_duration != original_duration:
                raise Exception("Reconstructed waveform has wrong duration.")

            return output

        else:
            # execute normally
            prediction = predictor({
                'waveform': waveform,
                'audio_id': ''})
            prediction.pop('audio_id')
        
            return prediction

    def separate_to_file(
            self, audio_descriptor, destination,
            audio_adapter=get_default_audio_adapter(),
            offset=0, duration=600., codec='wav', bitrate='128k',
            filename_format='{filename}/{instrument}.{codec}',
            synchronous=True):
        """ Performs source separation and export result to file using
        given audio adapter.

        Filename format should be a Python formattable string that could use
        following parameters : {instrument}, {filename} and {codec}.

        :param audio_descriptor:    Describe song to separate, used by audio
                                    adapter to retrieve and load audio data,
                                    in case of file based audio adapter, such
                                    descriptor would be a file path.
        :param destination:         Target directory to write output to.
        :param audio_adapter:       (Optional) Audio adapter to use for I/O.
        :param offset:              (Optional) Offset of loaded song.
        :param duration:            (Optional) Duration of loaded song.
        :param codec:               (Optional) Export codec.
        :param bitrate:             (Optional) Export bitrate.
        :param filename_format:     (Optional) Filename format.
        :param synchronous:         (Optional) True is should by synchronous.
        """
        waveform, _ = audio_adapter.load(
            audio_descriptor,
            offset=offset,
            duration=duration,
            sample_rate=self._sample_rate)

        if self.tf_session:
            with self.tf_session.as_default():
                with self.tf_session.graph.as_default():
                    sources = self.separate(waveform)
        else:
            sources = self.separate(waveform)

        filename = splitext(basename(audio_descriptor))[0]
        generated = []

        for instrument, data in sources.items():

            if instrument == 'vocals':
                path = join(destination, filename_format.format(
                    filename=filename,
                    instrument=instrument,
                    codec=codec))
                
                # override sample rate
                _sample_rate = self._sample_rate
                if self._pool:
                    task = self._pool.apply_async(audio_adapter.save, (
                        path,
                        data,
                        _sample_rate,
                        codec,
                        bitrate))
                    self._tasks.append(task)
                else:
                    audio_adapter.save(path, data, _sample_rate, codec, bitrate)
                
        if synchronous and self._pool:
            self.join()

        tf.reset_default_graph()
        tf.keras.backend.clear_session()

