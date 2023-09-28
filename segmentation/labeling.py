import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from utils.utils import plot_n_save


def full_stft(sig, sr):
    # Inputs:
    #   1. sig
    #   2. labels
    #
    # Outputs:
    #   1. stft - stft of signal

    # Time Resolution:
    # s1_maxtime = 0.15  # [seconds]
    # nperseg = sr * s1_maxtime
    nperseg = sr * 0.03  # Fixed 15ms resolution

    # Stft Calculation:
    f, t, stft = sp.signal.stft(sig, fs=sr, nperseg=nperseg)

    return f, t, stft


def choose_first_s1_time(f, t, stft, sr, stft_sr):
    # choose_first_s1:
    # Inputs:
    #   1. f, t, stft: return values of stft on signal
    #   2. sr: sample rate of signal
    # Outputs:
    #   1. s1_t: sample number of best s1

    # Define Mask:
    mask_tsize = int(np.ceil(150e-3 * stft_sr))
    mask_fsize = stft.shape[0]
    s1_flim = np.argmin(np.abs(f - 200))
    mask = np.ones((mask_fsize, mask_tsize))
    mask[s1_flim:, :] = -0.5

    # Loop over STFT:
    stft_amp = np.abs(stft)
    buffer = int(0.1 * stft.shape[1])
    max_val, max_sample = 0, 0
    for ti in np.arange(mask_tsize + buffer, stft.shape[1] - mask_tsize - buffer):
        current_val = np.sum(stft_amp[:, ti:ti + mask_tsize] * mask)
        if current_val > max_val:
            max_val = current_val
            max_sample = ti + mask_tsize // 2

    max_time = t[max_sample]
    max_sample = int(max_time * sr)

    return max_sample


def choose_first_s1_time_backup(f, t, stft, sr):
    # choose_first_s1:
    # Inputs:
    #   1. f, t, stft: return values of stft on signal
    #   2. sr: sample rate of signal
    # Outputs:
    #   1. s1_t: sample number of best s1

    # Find most dominant frequency:
    stft_amp = np.abs(stft)
    dominant_freq = np.argmax(np.sum(stft_amp, axis=1))

    # Find time frame:
    stft1d = stft_amp[dominant_freq, :]
    num_samples = stft1d.shape[0]
    buffer = int(0.1 * num_samples)
    stft1d = stft1d[buffer:-buffer]
    s1_t = np.argmax(stft1d)  # [samples in STFT domain]

    # Choose between adjacent peaks based on sys-dis diff:
    # max_indices = np.squeeze(sp.signal.argrelextrema(stft1d, np.greater, order=3))
    max_indices = np.squeeze(np.argwhere(stft1d - 0.6 * stft1d[s1_t] > 0))
    index_center = np.argmin(np.abs(s1_t - max_indices))
    max_indices = max_indices[np.abs(max_indices-max_indices[index_center] > 3)]
    index_center = np.argmin(np.abs(s1_t - max_indices))
    index_left = max(0, index_center - 1)
    index_right = min(max_indices.shape[0] - 1, index_center + 1)
    dis_samples = max_indices[index_center] - max_indices[index_left]
    sys_samples = max_indices[index_right] - max_indices[index_center]
    if sys_samples > dis_samples:
        s1_t = max_indices[index_left]

    s1_t = t[s1_t + buffer]  # [seconds]
    s1_t *= sr  # [samples]

    return int(s1_t)


def choose_first_s1_freq(sig, s1_t, sr):
    s1_maxtime = 0.15  # [seconds]
    s1_maxsamples = int(s1_maxtime * sr)  # [samples]
    offset = s1_maxsamples // 2
    wind = sig[s1_t - offset: s1_t + offset]
    wind_fft = sp.fft.fftshift(sp.fft.fft(wind))
    wind_fft_abs = np.abs(wind_fft)
    freq = sp.fft.fftshift(sp.fft.fftfreq(s1_maxsamples-1, 1/sr))
    s1_f = abs(freq[np.argmax(wind_fft_abs)])
    return s1_f


def define_window(sig, sr, left_bound, right_bound):
    # Defining window size:
    s1_maxtime = 0.15  # [seconds]
    s1_maxsamples = int(s1_maxtime * sr)  # [samples]
    s1_offset = s1_maxsamples // 2

    left_bound -= s1_offset
    right_bound += s1_offset

    if left_bound < 0 or right_bound >= sig.shape[0]:
        if left_bound < 0:
            window = np.pad(sig[0:right_bound], (np.abs(left_bound), 0), 'constant', constant_values=(0, 0))
        else:
            window = np.pad(sig[left_bound:], (0, np.abs(right_bound - sig.shape[0])), 'constant', constant_values=(0, 0))
    else:
        window = sig[left_bound:right_bound]

    win_size_samples = window.shape[0]
    return window, left_bound


def find_s1(f, t, stft_window, sr, s1_f):

    # Decimate:
    # window = sp.signal.decimate(window, 2)
    # sr = sr // 2
    # nperseg = int(0.03 * sr)
    # f, t, stft = sp.signal.stft(window, sr, nperseg=nperseg)

    # mask = build_mask(s1_f, f, t)
    #
    # mask_size = mask.shape[1]
    # stft_window = np.pad(stft_window, ((0, 0), (mask_size, mask_size)), 'constant', constant_values=(0, 0))
    #
    # stft_amp = np.abs(stft_window)
    #
    # max_val, max_time, max_sample = 0, 0, 0
    # for ti in range(stft_window.shape[1]-mask_size):
    #     current = np.sum(stft_amp[:, ti:ti + mask_size] * mask)
    #     if current > max_val:
    #         max_val = current
    #         max_time = (ti - mask_size / 2) * (1 / sr)  # [seconds]
    #         max_sample = int(max_time * sr)

    stft_amp = np.abs(stft_window)
    max_val, max_time, max_sample = 0, 0, 0
    for ti in range(stft_amp.shape[1]):
        current_val = np.sum(stft_amp[0:8, ti])
        if current_val > max_val:
            max_val = current_val
            max_time = ti * (1 / sr)  # [seconds]
            max_sample = int(max_time * sr)

    return max_sample


def build_mask(s1_f, f, t):

    # Defining mask gaussian parameters:
    mu_f = f[np.argmin(np.abs(f - s1_f))]
    mu_t = t[t.shape[0] // 2]
    sig_f = 3 * (f[1]-f[0])
    sig_t = 2 * (t[1]-t[0])

    # Defining mask:
    mask_f, mask_t = np.meshgrid(f, t)
    exp_f = (np.square(mask_f - mu_f)) / (2 * (sig_f ** 2))
    exp_t = (np.square(mask_t - mu_t)) / (2 * (sig_t ** 2))
    mask = np.exp(-exp_f) * np.exp(-exp_t)

    return mask.T


def define_stft_wind(stft, window_size, start_index, sr, t, stft_sr):

    # Calculate starting point in STFT:
    start_second = start_index / sr
    start_stftsample = max(int(np.floor(start_second * stft_sr))-1, 0)

    # Calculate Length in STFT:
    window_size_sec = window_size / sr
    window_size_stftsample = min(int(np.ceil(window_size_sec * stft_sr))+2, stft.shape[1] - 1)

    left_bound = start_stftsample
    right_bound = start_stftsample + window_size_stftsample
    if left_bound < 0 or right_bound >= stft.shape[1]:
        if left_bound < 0:
            stft_wind = np.pad(stft[:, 0:right_bound], ((0, 0), (np.abs(left_bound), 0)),
                               'constant', constant_values=(0, 0))
            t_wind = np.pad(t[0:right_bound], (np.abs(left_bound), 0),
                            'constant', constant_values=(0, 0))
        else:
            stft_wind = np.pad(stft[:, left_bound:], ((0, 0), (0, np.abs(right_bound - stft.shape[1]))),
                               'constant', constant_values=(0, 0))
            t_wind = np.pad(t[left_bound:], (0, np.abs(right_bound - t.shape[0])),
                            'constant', constant_values=(0, 0))
    else:
        t_wind = t[left_bound:right_bound]
        stft_wind = stft[:, left_bound:right_bound]

    return stft_wind, t_wind


def refine_time(sample, energy, sr, path):

    # Define window in Hilbert energy the size of 75ms:
    time_frame_seconds = 0.075  # sec
    time_frame = int(time_frame_seconds * sr)  # samples
    bound_left = max(0, sample - time_frame)
    bound_right = min(sample + 1, energy.shape[0])
    window = energy[bound_left:bound_right]

    # Find picks to the left of samples that are above th:
    th = 0.4 * energy[sample]
    indices = np.squeeze(np.argwhere(window >= th))
    refined_time = np.amin(indices) + bound_left

    return refined_time


def label_peaks(path, results_path, plot=False):
    # Inputs:
    #   1. path - string, path to raw data, sample rate, etc...
    # Outputs:
    #   1. stft - stft of signal

    # Load Data:
    sig = np.load(path + 'data.npy')
    labels = sig[1, :, :]
    sig = sig[0, :, :]
    sig = np.reshape(sig, -1)
    labels_shape = labels.shape
    labels = np.reshape(labels, -1)
    sr = np.load(path + 'sr.npy')
    hr = np.load(path + 'hrEstimate.npy')
    energy = np.reshape(np.load(path + 'he.npy'), -1)

    # # Remove Leading and Trailing Zeros From Calculation:
    # sig_b = np.trim_zeros(sig, 'b')
    # zeros_b = sig.shape[0] - sig_b.shape[0]
    # sig_f = np.trim_zeros(sig, 'f')
    # zeros_f = sig.shape[0] - sig_f.shape[0]
    # sig = np.trim_zeros(sig)
    # if zeros_b > 0:
    #     energy = energy[0:-zeros_b]
    # if zeros_f > 0:
    #     energy = energy[zeros_f:]

    energy_num = energy.shape[0]
    time_axis = np.linspace(0, (1 / sr) * energy_num, energy_num)

    # STFT:
    f, t, stft = full_stft(sig, sr)
    stft_sr = 1 / (t[1]-t[0])

    # Find First S1 Sound:
    s1_t = choose_first_s1_time(f, t, stft, sr, stft_sr)
    s1_t = refine_time(s1_t, energy, sr, path)
    s1_f = choose_first_s1_freq(sig, s1_t, sr)

    bpm_error = 5
    current_index, step_left, step_right = s1_t, int(sr / ((hr+bpm_error) / 60)), int(sr / ((hr-bpm_error) / 60))
    step_size = (step_right + step_left) // 2
    guesses_right = [s1_t]
    windows_left = []
    s1_maxtime = 0.15  # [seconds]
    s1_bound = int(s1_maxtime * sr / 2)  # [samples]

    # Label peaks to the right
    while current_index < sig.shape[0] - step_right:
        window, start_index = define_window(sig, sr, current_index+step_left, current_index+step_right)

        stft_window, t_wind = define_stft_wind(stft, window.shape[0], start_index, sr, t, stft_sr)
        if np.count_nonzero(t_wind) == 0:
            break

        s1_sample = find_s1(f, t_wind, stft_window, stft_sr, s1_f)
        s1_sample = int(t_wind[s1_sample] * sr)
        s1_sample = refine_time(s1_sample, energy, sr, path)
        guesses_right.append(s1_sample)
        windows_left.append(start_index)
        current_index = s1_sample

    # Label peaks to the left:
    current_index = s1_t
    # if path.split('/')[-2] == '09_45_01_input_090_bme.wav':
    #     print('hi')
    while current_index > step_left:
        window, start_index = define_window(sig, sr, current_index-step_right, current_index-step_left)
        # Swapped right and left from the above loop to make sure left_bound < right_bound

        stft_window, t_wind = define_stft_wind(stft, window.shape[0], start_index, sr, t, stft_sr)
        if np.count_nonzero(t_wind) == 0:
            break

        s1_sample = find_s1(f, t_wind, stft_window, stft_sr, s1_f)
        s1_sample = int(t_wind[s1_sample] * sr)
        s1_sample = refine_time(s1_sample, energy, sr, path)
        guesses_right.append(s1_sample)
        windows_left.append(start_index)
        current_index = s1_sample

    guesses_right = np.asarray(guesses_right, dtype='float')

    guesses = np.array(np.sort(guesses_right), dtype='int')
    np.save(path + 'guesses.npy', guesses)

    if plot:
        plot_n_save(sig, sr, 'S1', results_path, guesses)

    return guesses


def guess_dataset(data_path, results_path):
    
    directory = os.fsencode(data_path)
    index = 0
    print("\n#####################\nFinding S1 locations for each recording: \n")

    for file in tqdm(os.listdir(directory), desc='Labeling: ', unit='recording'):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            _ = label_peaks(data_path + filename + '/', results_path + '2_s1locs/' + str(index), plot=True)
            index += 1
    
    print("\n Done labeling!")
    return
