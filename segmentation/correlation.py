import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from segmentation.envelope import *
import os
from tqdm import tqdm


def estimate_heartrate(path, plot=False):
    # Inputs:
    #   1. path - path to wav (data, labels, sr, etc..)
    # Outputs:
    #   1. estimated_hr - estimated heartrate based on auto-correlation

    # Calculate Energy:
    energy = calc_envelope(path)
    np.save(path+'he.npy', energy)
    energy = np.reshape(energy, -1)
    energy = np.trim_zeros(energy)
    sr = np.load(path+'sr.npy')

    # Samples boundaries:
    maximum_heartrate = 150  # bpm
    maximum_heartrate /= 60  # bps
    dmin = int(sr / maximum_heartrate)  # minimum number of samples
    minimum_heartrate = 40   # bpm
    minimum_heartrate /= 60  # bps
    dmax = int(sr / minimum_heartrate)  # maximum number of samples

    # Auto Correlation:
    corr = sp.signal.correlate(energy, energy, mode='same')
    np.save(path + 'corr.npy', corr)

    # Find indices of maximums:
    # peaks_indices = np.squeeze(np.asarray(signal.argrelextrema(corr, np.greater, order=dmin)))
    center = corr.shape[0] // 2
    d = np.argmax(corr[center + dmin: center + dmax]) + dmin
    estimated_hr = (60 * sr) / d

    # For Debugging, Plot Energy and Correlation:
    if plot:
        # Print only very wrong guesses:
        ground_truth = np.load(path + 'hrSanolla.npy')
        # if np.abs(estimated_hr - ground_truth) > 5:
        t_axis = np.arange(0, energy.shape[0] / sr, 1 / sr)
        sanolla_d = int((60 * sr) / ground_truth)
        fig, ax = plt.subplots(2, 1)
        # ax[0].plot(t_axis, energy)
        # ax[1].plot(t_axis, corr)
        # ax[1].scatter(t_axis[peaks_indices], corr[peaks_indices])
        ax[0].plot(energy)
        ax[1].plot(corr)
        ax[1].plot(np.array([center + dmin, center + dmin]), np.array([np.amax(corr), np.amin(corr)]), 'r--')
        ax[1].plot(np.array([center + dmax, center + dmax]), np.array([np.amax(corr), np.amin(corr)]), 'r--', label='Window')
        ax[1].scatter(center + d, corr[center + d], color='r', label='Guess')
        if sanolla_d > 0:
            ax[1].scatter(center + sanolla_d, corr[center + sanolla_d], color='g', label='Sanolla')
        ax[1].legend()
        plt.suptitle(path.split('/')[2])
        ax[0].text(10, np.amax(energy) - 0.1, 'Sanolla: ' + str(ground_truth) + ', Guess: ' + str(estimated_hr))
        plt.show()
    return estimated_hr


def estimate_dataset(data_path, plot=False):
    
    file_names = []
    estimated_hrs = []
    sanolla_hrs = []
    
    print("\n#####################\nEstimating Heartrate for each recording: \n")

    for file in tqdm(os.listdir(data_path), desc='Estimating: ', unit='recording'):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            file_names.append(filename)
            current_estimate = estimate_heartrate(data_path + filename + '/', plot=plot)
            estimated_hrs.append(current_estimate)
            np.save(data_path + filename + '/hrEstimate.npy', current_estimate)

    print("\n Done estimating!")
    return
    