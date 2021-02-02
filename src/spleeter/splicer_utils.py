import numpy as np



def get_duration(y, sr):
    import librosa
    duration = librosa.get_duration(y=y, sr=sr)

    return duration


def get_splices(y, sr, splices_duration, orig_duration):
    
    splices = {}
    
    ch1 = y[:, 0]
    ch2 = y[:, 1]
    
    splits = orig_duration/splices_duration
    int_splits = int(round(splits))
    if splits > int_splits:
        splits = int_splits + 1
    else:
        splits = int_splits
    
    for i in range(0, splits):
        
        if i == 0:
            
            signal_1 = ch1[ : splices_duration * sr ]
            signal_2 = ch2[ : splices_duration * sr ]
            signal = np.array([signal_1, signal_2])
            
        else:
            if i != splits:
                
                start = splices_duration * i
                end = start + splices_duration
                
                signal_1 = ch1[ start * sr : end * sr ]
                signal_2 = ch2[ start * sr : end * sr ]
                
                signal = np.array([signal_1, signal_2])
            
            else:
                start = splices_duration * i
                
                signal_1 = ch1[ start * sr : ]
                signal_2 = ch2[ start * sr : ]
                
                signal = np.array([signal_1, signal_2])
        
        signal = np.swapaxes(signal,0,1)
        splices[i] = signal

    return splices


def reconstruct_waveform(splices):
    
    waveform = []
    
    for i, val in splices.items():
        waveform.extend(val)
    
    waveform = np.array(waveform)
    
    return waveform


def splicer(y, sr, splices_duration=30):

    audio_duration = get_duration(y[:,0], sr)

    if audio_duration <= splices_duration:
        output = y
    else:
        splices = get_splices(y, sr, splices_duration, audio_duration)
        output = splices

    return output