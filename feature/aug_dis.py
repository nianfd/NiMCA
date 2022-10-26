import librosa
import random
import numpy as np
import os
import re
import shutil
from scipy.signal import butter, lfilter, freqz
min_length=0.02
mid_length=0.2
max_length=0.4
random.seed(100)
np.random.seed(200)
def get_wav_path(wav_dir, audio_list):
    pattern = re.compile(r'^[^.].*\.wav$')
    for directory, filedir, filenames in os.walk(wav_dir):
        for filename in filenames:
            if pattern.match(filename):
                audio_list.append(os.path.join(directory,filename))
    return audio_list

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y  # Filter requirements.
def cross_fade_in_fade_out(silence_len):
    fade_len = silence_len
    silence = np.zeros((silence_len), dtype=np.float64)
    linear = np.ones((silence_len), dtype=np.float64)
    t = np.linspace(-1, 1, fade_len, dtype=np.float64)
    fade_in = 0.5 * (1 + t)
    fade_out = 0.5 * (1 - t)
    # Concat the silence to the fades
    fade_in = np.concatenate([silence, fade_in])
    fade_out = np.concatenate([linear, fade_out])
    return fade_in, fade_out

def process_dis(wav_path, dst_path):
    y,sr=librosa.load(wav_path,sr=44100)
    y=y[:sr*10]
    cross_fade_len = 440
    start_length=random.randint(cross_fade_len, int(len(y)/2)-cross_fade_len)
    end_length=start_length
    count=0


    rand=False
    last_rand=False
    dis=False
    dis_count=0
    while end_length<len(y)-cross_fade_len*20:
        if random.random()>0.999:
            random_length = random.randint(min_length*sr, mid_length*sr)
            if random.random()>0.5 and count < 15:
                rand=True
                count += 1
        else:
            random_length = random.randint(mid_length * sr, max_length * sr)
            if random.random()>0.5 and count < 15:
                rand = True
                count += 2

        end_length = start_length + random_length
        if end_length > len(y):
            end_length = len(y)-cross_fade_len

        if not last_rand and rand:
            dither=np.random.normal(loc=0, scale=0.01, size=end_length - start_length+cross_fade_len*2) * 0.1
            dither1 = butter_lowpass_filter(dither, 150, fs=sr, order=5)+dither*0.01

            fade_in, fade_out = cross_fade_in_fade_out(int(cross_fade_len/2))
            y[start_length-cross_fade_len:start_length]= y[start_length-cross_fade_len:start_length]*fade_out+ dither1[0:cross_fade_len]*fade_in
            try:
                y[end_length:end_length+cross_fade_len] = y[end_length:end_length+cross_fade_len] * fade_in + dither1[-cross_fade_len:]*fade_out
            except:
                import sys
                print(end_length, end_length+cross_fade_len)
                print(-cross_fade_len)
                return False, 0
            #y[end_length:end_length+cross_fade_len] = y[end_length:end_length+cross_fade_len] * fade_in + dither1[-cross_fade_len:]*fade_out
            y[start_length:end_length] = dither1[cross_fade_len:-cross_fade_len]
            rand=False
            last_rand=True
            dis =True
            dis_count +=1
        else:
            last_rand =False
        start_length = end_length
    if dis and dis_count>5:
        librosa.output.write_wav(dst_path, y, sr)
    return dis, dis_count

def create_dis(wav_list,count, save_dir):
    with open('label_aug_momo_scr1_dis_2.scp', 'w') as f:
        for i in range(2000):
            if count<1000:
                wav_path=wav_list[i]
                print(wav_path)
                filename=os.path.basename(wav_path)
                newfilename='momo_sqa_aug_scr1_dis_'+filename.split('_')[-1]
                savefilename=os.path.join(save_dir, newfilename)
                dis, dis_count=process_dis(wav_path, savefilename)
                if dis and dis_count>5:
                    f.write(newfilename +' 1 -1 -1 -1 -1'+'\n')
                    count+=1 
                    print(dis_count)
    print('count:',count)

def mv_audio(wav_list, dst_dir):
    for i in range(1000):
        wav_path=wav_list[i]
        basename=os.path.basename(wav_path)
        dst_path=os.path.join(dst_dir,basename)
        shutil.copy(wav_path,dst_path) 
if __name__=='__main__':
    dst_dir='need_vol_aug_batch2'
    save_dir='data_aug_1000_scr1_dis_2'
    wav_dir=['data1_1572_441k ','data2_3000_441k','data3_1481_441k','data4_2870']
    wav_list=[]
    for i in wav_dir:
        wav_list=get_wav_path(i, wav_list)
    count=0
    random.shuffle(wav_list)
#    create_dis(wav_list,count,save_dir)
    mv_audio(wav_list, dst_dir)
