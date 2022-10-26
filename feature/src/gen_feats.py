import librosa as lb
import numpy as np
import sys
import os

def get_librosa_melspec(file_path):
    # Calc spec
    y, sr = lb.load(file_path, sr=44100)
    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=1024,
        hop_length=441,
        win_length=None,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=1.0,

        n_mels=48,
        fmin=0.0,
        fmax=20e3,
        htk=False,
        norm='slaney',
        )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)

    return spec



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('{} src_dir dst_dir'.format(sys.argv[0]))
        exit(0)
    for fn in os.listdir(sys.argv[1]):
        fname = os.path.join(sys.argv[1], fn)
        mel_spec = get_librosa_melspec(fname).T
        oname = os.path.join(sys.argv[2], fn.replace('.wav', '.npy'))
        print(oname)
        np.save(oname, mel_spec)
