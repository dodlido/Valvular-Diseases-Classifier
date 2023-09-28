
import numpy as np
import scipy
import torchaudio
from scipy import signal
from scipy import ndimage
from utils.utils import plot_n_save


def segment_wav(wav_path, ws, sanolla):
    # segment_wav:
    # inputs:
    #   1. wav_path (string) - path to wav file to read
    #   2. ws (float) - window size in seconds
    #   3. throwaway_frac (2,) ndarray - fraction of samples to throw away from the start and end of signal
    # outputs:
    #   1. windows (ndarray) - windows in wav, shape (windows_num, window_samples)
    #   2. sr (int) - sample rate of wav
    sr, sig = scipy.io.wavfile.read(wav_path, mmap=False)
    sig = np.squeeze(np.array(sig))
    if sanolla:
        sig = sig[sr:-sr]
    samples = sig.shape[0]
    window_samples = int(ws * sr)
    windows_num = int(np.ceil(samples/window_samples))
    pad_num = window_samples * windows_num - samples
    sig = np.pad(sig, (0, pad_num), 'constant', constant_values=(0, 0))
    windows = np.reshape(sig, (windows_num, window_samples))
    return windows.astype(float), sr


def norm_mean(windows):
    # norm_mean:
    # inputs:
    #   1. windows (ndarray) - windows in wav, shape (windows_num, window_samples)
    # outputs:
    #   1. windows (ndarray) - normalized windows in wav, shape (windows_num, window_samples)
    mean_sig = windows.mean()
    windows -= mean_sig
    max_sig = np.amax(np.abs(windows))
    windows /= max_sig
    return windows


def remove_invalid(windows, sr):
    # remove_invalid:
    # inputs:
    #   1. windows (ndarray) - normalized windows in wav, shape (windows_num, window_samples)
    #   2. sr (int) - sample rate of wav
    # outputs:
    #   1. windows (ndarray) - normalized valid windows in wav, shape (valid_num, window_samples)
    #   2. wn (int) - number of valid windows
    num_sat = np.count_nonzero(windows >= 1, axis=1)
    valid = np.zeros_like(num_sat)
    valid[num_sat <= (0.5 * sr)] = 1
    wn = np.count_nonzero(valid)
    valid_indices = np.argwhere(valid != 0)
    windows = np.delete(windows, np.argwhere(valid == 0), axis=0)
    return windows, wn, valid_indices


def filter_windows(windows, orig_sr, target_sr, bp_freq, median_ker, save_path, results_path, plot=False):
    # remove_invalid:
    # inputs:
    #   1. windows (ndarray) - normalized windows in wav, shape (windows_num, window_samples)
    #   2. orig_sr (int) - sample rate of wav
    #   3. targe_sr (int) - target sample rate
    #   4. bp_freq (list) - shape (,2) position 1 is pass freq, position 2 is stop freq [Hz]
    #   5. median_ker (int) - size of median_filter kernel
    # outputs:
    #   1. windows (ndarray) - normalized valid windows in wav, shape (valid_num, window_samples)
    #   2. achieved_sr (int) - achieved sample rate
    nf = orig_sr / 2
    deci_fac = int(np.floor(orig_sr/target_sr))
    achieved_sr = int(np.floor(orig_sr / deci_fac))
    bp_freq = np.array(bp_freq).astype('float64')
    bp_freq /= nf
    sos = signal.butter(8, bp_freq, btype='bandpass', output='sos')
    plot_n_save(windows, orig_sr, 'Raw', results_path)
    windows = signal.sosfilt(sos, windows, axis=1)
    plot_n_save(windows, orig_sr, 'BP', results_path)
    np.save(save_path + '/bp.npy', windows)
    windows = signal.decimate(windows, deci_fac, axis=1)
    plot_n_save(windows, target_sr, 'Deci', results_path)
    np.save(save_path + '/deci.npy', windows)
    windows = ndimage.median_filter(windows, size=(1, median_ker))
    np.save(save_path + '/median.npy', windows)
    if plot:
        plot_n_save(windows, target_sr, 'Medi', results_path)
    return windows, achieved_sr
