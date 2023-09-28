import os.path

from python_speech_features import mfcc
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from skimage import io
from pyAudioAnalysis import ShortTermFeatures as stf
from torchvision import transforms
import torch
import librosa, librosa.display
from PIL import Image
import csv
from tqdm import tqdm
from utils.utils import *


def calculate_flux(sig, sr):
    # divide the segmented signal length by half
    fft_frame_length = len(sig) / 2
    # extract the signal by half
    first_frame = sig[:int(fft_frame_length)]
    second_frame = sig[int(fft_frame_length):]
    frame_step = 1
    while (first_frame.shape != second_frame.shape):
        first_frame = sig[:frame_step + int(fft_frame_length)]
        second_frame = sig[int(fft_frame_length):]
        frame_step = frame_step + 1

    # calculate the fft of the signal
    fft_first_frame = np.array([np.fft.fft(first_frame)])
    fft_second_frame = np.array([np.fft.fft(second_frame)])
    # extract the spectral flux features
    spectral_flux = np.array(stf.spectral_flux(np.abs(fft_first_frame), np.abs(fft_second_frame)))

    return spectral_flux


def calculate_mean_freq(sig, sr):
    frequency_domain = np.array([np.fft.fft(sig)])
    # calculate mean
    mean_frequency_domain = np.mean(frequency_domain)
    # calculate standard deviation
    std_frequency_domain = np.std(frequency_domain)
    # extract the real and the imaginary number from complex number
    mean_frequency_domain_real = mean_frequency_domain.real
    mean_frequency_domain_imaginary = mean_frequency_domain.imag

    return mean_frequency_domain_real, mean_frequency_domain_imaginary


def extract_manual_features(cycles_path, features_path, featype_paths, sr, results_path):

    print("\n#####################\nExtracting manual features: \n")

    dataset = np.load(cycles_path + 'dataset.npz')
    data, padnum, labels = dataset['data'], dataset['padnum'], dataset['labels'] 

    with open(features_path + featype_paths + 'features.csv', 'w', newline='') as file:
        
        writer = csv.writer(file)
        field = ["cycle_#", "zero_cross", "mean_mfcc", "std_mfcc", "spectral_centroid_1",
                 "spectral_centroid_2", "spectral_centroid_3", "spectral_rolloff_1",
                 "spectral_rolloff_2", "spectral_rolloff_3", "spectral_flux", "mean_frequency_real",
                 "mean_frequency_imaginary", "energy_entropy", "pad_num", "label"]
        writer.writerow(field)

        for i in tqdm(range(data.shape[0]), desc='Extracting', unit='cycle'):
            sig = data[i, :]
            zero_crossing_rate = stf.zero_crossing_rate(sig)
            mfcc = librosa.feature.mfcc(y=sig.astype('float32'), sr=sr)
            mean_mfcc = np.mean(mfcc)
            std_mfcc = np.std(mfcc)
            spectral_centroid = np.squeeze(librosa.feature.spectral_centroid(y=sig, sr=sr))
            spectral_rolloff = np.squeeze(librosa.feature.spectral_rolloff(y=sig, sr=sr))
            spectral_flux = calculate_flux(sig, sr)
            mean_frequency_domain_real, mean_frequency_domain_imaginary = calculate_mean_freq(sig, sr)
            energy_entropy = np.array(stf.energy_entropy(sig))

            row = np.array([i, zero_crossing_rate, mean_mfcc, std_mfcc, spectral_centroid[0], spectral_centroid[1],
                           spectral_centroid[2], spectral_rolloff[0], spectral_rolloff[1], spectral_rolloff[2],
                            spectral_flux, mean_frequency_domain_real, mean_frequency_domain_imaginary,
                            energy_entropy, padnum[i], labels[i]])
            writer.writerow(row)

    visualize_all_features_1d(field[1:-1], features_path + featype_paths + 'features.csv', results_path + featype_paths)
    
    print("\n Manual features extracted!")

    return


def mel_spectogram_single(signal, sr, padnum, fmax=500):
    
    n_fft = 64

    if padnum==0:
        y = signal
    else:
        y = signal[0:-padnum]

    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13, n_fft=n_fft, win_length=n_fft,
                                            fmax=fmax, hop_length=int(n_fft//4))
    spect_dB = librosa.power_to_db(spect, ref=np.max)
    # spect_dB = spect
    
    spect_dt = np.gradient(spect_dB, axis=1)
    spect_dt2 = np.gradient(spect_dt, axis=1)

    mel_single = np.zeros((3, spect_dB.shape[0], spect_dB.shape[1]))
    mel_single[0, :, :], mel_single[1, :, :], mel_single[2, :, :] = spect_dB, spect_dt, spect_dt2

    resize_mel = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 128), antialias=True),])
    # resize_mel = transforms.Compose([transforms.ToTensor(),])

    mel_single = np.moveaxis(mel_single, source=0, destination=2)
    mel_single = (mel_single)
    mel_single = resize_mel(mel_single).numpy()
    mel_single = np.moveaxis(mel_single, source=0, destination=2)
    mel_single -= mel_single.max()
    mel_single = np.abs(mel_single)
    mel_single *= (255.0/mel_single.max())
    mel_single = mel_single.astype('uint8')
        
    return mel_single


def extract_mel_features(cycles_path, features_path, featype_paths, sr, results_path, plot=False):
    
    print("\n#####################\nExtracting Mel-images features: \n")
    
    dataset = np.load(cycles_path + 'dataset.npz')
    cycles, hr, padnum, labels, ids = dataset['data'], dataset['hr'], dataset['padnum'], dataset['labels'], dataset['ids']
    cycles_num = cycles.shape[0]

    with open(features_path + featype_paths[0] + 'melimgs.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["cycle_#", "patient_id", "time", "hr", "padnum", "label"]
        writer.writerow(field)

        index = 1

        for i in tqdm(range(cycles_num), desc='Extracting: ', unit='cycle'):
            
            if labels[i]!=-1:

                current_melImg = np.expand_dims(mel_spectogram_single(signal=cycles[i, :], padnum=padnum[i], sr=sr), 0)
                if plot:
                    save_spectogram(np.squeeze(current_melImg)[:, :, 0], results_path + featype_paths[0] + str(index) + '/')
                
                if i==0:
                    feature_maps_array = current_melImg
                else:
                    feature_maps_array = np.append(feature_maps_array, current_melImg, axis=0)

                row = np.array([index, ids[i,0], ids[i,1], hr[i], padnum[i], labels[i]])
                writer.writerow(row)          
                
                index += 1         
    
    np.save(features_path + featype_paths[0] + 'melimgs.npy', feature_maps_array)

    print("\n Mel-images extracted!")
    print("\n#####################\nExtracting Mel-vectors features: \n")

    time_samples, freq_samples = 100, 13

    with open(features_path + featype_paths[1] + 'melvectors.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["cycle_#", "patient_id", "time", "hr", "padnum", "label"]
        for i in range(time_samples * freq_samples):
            field.append("s_"+str(i))
        writer.writerow(field)

        for i in tqdm(range(feature_maps_array.shape[0]), desc='Extracting', unit='cycle'):
        
            current_melImg = feature_maps_array[i, :, :, :]
            time_length = current_melImg.shape[1]
            indices_to_sample = np.linspace(start=0, stop=time_length-1, num=time_samples).astype('int')
            current_feature_vector = current_melImg[:, indices_to_sample, 0]
            current_feature_vector = np.reshape(current_feature_vector, -1)
            current_feature_vector = np.expand_dims(current_feature_vector, 0)
    
            row = np.array([index, ids[i,0], ids[i,1], hr[i], padnum[i], labels[i]])
            row = np.append(row, np.squeeze(current_feature_vector), axis=0)
            writer.writerow(row)          

    print("\n Mel-vectors extracted!")

    return


def extract_features_top(cycles_path, features_path, featype_paths, sr, results_path):

    # Extract manual features:
    extract_manual_features(cycles_path, features_path, featype_paths[0], sr, results_path + '3_features/')
    
    # Extract mel images and vectors:
    extract_mel_features(cycles_path, features_path, featype_paths[1:], sr, results_path + '3_features/', plot=True)

    return
