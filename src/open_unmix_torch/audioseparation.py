import os
import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy
import warnings
from contextlib import redirect_stderr
import io
import datetime

from . import model_unmix
from . import utils_unmix

samplerate = 44100

# Define Paths
MXM_ROOT = os.getenv('MXM_ROOT', '/root')
OUTDIR = os.path.join(MXM_ROOT, 'audio')


def get_models(targets, model_name, device):
    
    unmix_target_list = []

    for j, target in enumerate(targets):
        unmix_target = load_model(
            target=target,
            model_name=model_name,
            device=device
        )
        unmix_target_list.append(unmix_target)

    return unmix_target_list


def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()

    if not model_path.exists():
        # model path does not exist, use hubconf model
        print('***** MODEL PATH DOES NOT EXIST, LOADING MODEL FROM TORCH.HUB')
        try:
            # disable progress bar
            err = io.StringIO()
            with redirect_stderr(err):
                return torch.hub.load(
                    'sigsep/open-unmix-pytorch',
                    model_name,
                    target=target,
                    device=device,
                    pretrained=True
                )
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
            # assume model is a path to a local model_name direcotry
    else:
        # load model from disk
        with open(Path(model_path, 'training-json-logs', target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )

        max_bin = utils_unmix.bandwidth_to_max_bin(
            state['sample_rate'],
            results['args']['nfft'],
            results['args']['bandwidth']
        )

        unmix = model_unmix.OpenUnmix(
            n_fft=results['args']['nfft'],
            n_hop=results['args']['nhop'],
            nb_channels=results['args']['nb_channels'],
            hidden_size=results['args']['hidden_size'],
            max_bin=max_bin
        )

        unmix.load_state_dict(state)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)
        return unmix


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def audio_preprocessing(input_file):
    # handling an input audio path
    audio, rate = sf.read(input_file, always_2d=True)

    if audio.shape[1] > 2:
        warnings.warn(
            'Channel count > 2! '
            'Only the first two channels will be processed!')
        audio = audio[:, :2]

    if rate != samplerate:
        # resample to model samplerate if needed
        audio = resampy.resample(audio, rate, samplerate, axis=0)

    if audio.shape[1] == 1:
        # if we have mono, let's duplicate it
        # as the input of OpenUnmix is always stereo
        audio = np.repeat(audio, 2, axis=1)

    return audio


def separate(audio, targets, unmix_target_list, niter=1, softmask=False, alpha=1.0, residual_model=False, device='cpu'):

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []

    for j, target in enumerate(targets):
        unmix_target = unmix_target_list[j]
        Vj = unmix_target(audio_torch).cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]
    
    V = np.transpose(np.array(V), (1, 3, 2, 0))

    X = unmix_target.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(targets) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=unmix_target.stft.n_fft,
            n_hopsize=unmix_target.stft.n_hop
        )
        estimates[name] = audio_hat.T

    return estimates


def process(input_file, targets, unmix_target_list):
    
    # audio preprocessing
    audio = audio_preprocessing(input_file)

    # separation
    estimates = separate(audio, targets, unmix_target_list)

    head, tail = os.path.split(input_file)
    filename, file_extension = os.path.splitext(tail)
    # write out estimates
    for target, estimate in estimates.items():
        if target in ['vocals', 'drums', 'bass', 'other']:
            output_file = os.path.join(OUTDIR, filename + '_' + target + '.wav')
            sf.write(
                output_file,
                estimate,
                samplerate
            )
    return
