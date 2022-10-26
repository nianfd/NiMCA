
import os
import sys
import numpy as np
import librosa


def get_librosa_melspec(file_path):
    # Calc spec
    y, sr = librosa.load(file_path, sr=44100)
    S = librosa.feature.melspectrogram(
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

    spec = librosa.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    return spec


def get_noise_lps_multiband(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    S = np.abs(librosa.stft(y=y,n_fft=1024,hop_length=441,win_length=None,window='hann',center=True,pad_mode='reflect'))
    tmp1 = S ** 2
    S2mb1 = np.sum(tmp1[1:129, :], axis=0)
    S2mb2 = np.sum(tmp1[129:257, :], axis=0)
    S2mb3 = np.sum(tmp1[257:385, :], axis=0)
    S2mb4 = np.sum(tmp1[385:513, :], axis=0)
    tmp2 = np.vstack((S2mb1, S2mb2, S2mb3, S2mb4))
    S2 = librosa.power_to_db(tmp2)
    return S2


def get_noisy_lps_multiband(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    S = np.abs(librosa.stft(y=y,n_fft=1024,hop_length=441,win_length=None,window='hann',center=True,pad_mode='reflect'))
    tmp1 = S ** 2
    S2mb1 = np.sum(tmp1[  1:129, :], axis=0)
    S2mb2 = np.sum(tmp1[129:257, :], axis=0)
    S2mb3 = np.sum(tmp1[257:385, :], axis=0)
    S2mb4 = np.sum(tmp1[385:513, :], axis=0)
    tmp2 = np.vstack((S2mb1, S2mb2, S2mb3, S2mb4))
    S2 = librosa.power_to_db(tmp2)
    return S2


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('{} src_dir1 src_dir2 dst_dir'.format(sys.argv[0]))
        exit(0)
    for fn in os.listdir(sys.argv[1]):
        try:
            fname1 = os.path.join(sys.argv[1], fn)
            #tmp = fn.replace('.wav', '_noise.wav')
            #fname2 = os.path.join(sys.argv[2], tmp)
            mel_spec = get_librosa_melspec(fname1).T
            noisy_lps = get_noisy_lps_multiband(fname1).T
            #noise_lps = get_noise_lps_multiband(fname2).T
            oname = os.path.join(sys.argv[3], fn.replace('.wav', '.npy'))
#            feat_combine = np.concatenate((mel_spec, noisy_lps,noisy_lps), axis=1)
            feat_combine = np.concatenate((mel_spec, noisy_lps), axis=1)
            print(oname)
            np.save(oname, feat_combine)
        except Exception as e:
            print("error, {}".format(e))
            pass    
