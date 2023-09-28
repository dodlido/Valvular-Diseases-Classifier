import numpy as np
import os
from scipy import interpolate
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def load_data(wav_path):
    sig = np.load(wav_path + 'data.npy')
    sig = sig[0, :, :]
    sig = np.reshape(sig, -1)
    labels = np.reshape(np.load(wav_path + 'guesses.npy'), -1)
    sr = np.load(wav_path + 'sr.npy')
    hr = int(np.load(wav_path + 'hrEstimate.npy'))
    tag = int(np.load(wav_path + 'diagnosis.npy'))
    return sig, labels, sr, hr, tag


def in_tolerance(segment, sr, hr, tol):
    # in_tolerance:
    #   Inputs:
    #       1. segment - segment to check
    #       2. sr - sample rate [Hz]
    #       3. hr - estimated heart rate [bpm]
    #       4. tol - percentage of heartrate to consider [0-1]
    #   Outputs:
    #       1. inside - indicates that a segment is inside the desired tolerance
    inside = 0
    samples = segment.shape[0]
    time = samples / sr
    bps = hr / 60
    bps_tol = bps * np.array([1-tol, 1+tol])
    if bps_tol[0] < time < bps_tol[1]:
        inside = 1
    return inside


def check_zero(segment, th):
    mean = np.mean(np.abs(segment))
    not_zero = 0
    if mean > th:
        not_zero = 1
    return not_zero


def valid_segment(segment, sr, hr, bpm_tol, zero_th):
    inside = in_tolerance(segment, sr, hr, bpm_tol)
    not_zero = check_zero(segment, zero_th)
    return inside * not_zero


def is_correct(segment_bounds, labels, correctness_tol, sr):
    # Inputs:
    #   1. segment_bounds - tuple, (start index, end index)
    #   2. labels - ndarray, size of sig, 1 where sanolla labeled S1 sound
    #   3. correctness_tol - distance between label and guess which is still correct [ms]
    #   4. sr - sample rate [Hz]
    # Outputs:
    #   1. correct - 1/0, 1 if segment was guessed correctly
    start_ok, end_ok = 0, 0
    labels_indices = np.argwhere(labels > 0)
    closest_start = np.amin(np.abs(labels_indices - segment_bounds[0]))
    closest_end = np.amin(np.abs(labels_indices - segment_bounds[1]))
    tol = (correctness_tol / 1000) * sr  # [samples]
    if closest_start < tol:
        start_ok = 1
    if closest_end < tol:
        end_ok = 1
    correct = start_ok * end_ok
    return correct


def list_valid_segments(sig, guesses, labels, sr, hr, bpm_tol, zero_th, correctness_tol):
    data, correct = [], []
    n_segments = 0
    for seg in range(guesses.shape[0] - 1):
        current_segment = sig[guesses[seg]:guesses[seg + 1]]
        if valid_segment(current_segment, sr, hr, bpm_tol, zero_th):
            data.append(current_segment)
            correct.append(is_correct([guesses[seg], guesses[seg + 1]], labels, correctness_tol, sr))
            n_segments += 1
    return data, correct, n_segments


def list_segments(sig, labels, sr, bpm_max, bpm_min):
    data, n_segments = [], 0
    bps_max, bps_min = bpm_max / 60, bpm_min / 60
    samples_min, samples_max = sr / bps_max, sr / bps_min
    if labels.shape[0] > 1:
        for seg in range(labels.shape[0]-1):
            current_seg = sig[labels[seg]:labels[seg+1]]
            if samples_min < current_seg.shape[0] < samples_max:
                data.append(current_seg)
                n_segments += 1
    return data, n_segments


def read_single(wav_path):
    # read_single:
    #   Inputs:
    #       1. wav_path - wav path to segment
    #   Outputs:
    #       1. n_segments - number of valid segments found
    #       2. data - a 2d list of (n_segments, segment_length) dimensions
    #       3. correct - a list of (n_segments) that signifies that the segment was labeled correctly

    # Load Data from path:
    sig, labels, sr, hr, tag = load_data(wav_path)

    # List valid segments:
    # bpm_tol, zero_th, correctness_tol = 0.4, 1e-3, 70
    data, n_segments = list_segments(sig, labels, sr, bpm_max=140, bpm_min=40)

    return n_segments, data, hr, tag


def list_all_segments(data_path):
    # List all segments in path:
    n_segments = 0
    data, correct, hrs, labels, ids = [], [], [], [], []

    dataset_name = data_path.split('/')[-2]

    for file in os.listdir(data_path):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            current_segments, current_data, hr, tag = read_single(data_path + filename + '/')
            s = filename.split('_')
            if dataset_name == '1_sanollav1':
                current_patient, current_time = int(s[0]), int(s[1] + s[2] + s[3])
            elif dataset_name == '2_sanollav2':
                current_patient, current_time = int(s[4]), int(s[0] + s[1] + s[2])
            elif dataset_name == '3_peterj':
                current_patient, current_time = int(s[0]), float(s[1])/10000
            else:
                print("Invalid dataset name!")
                return -1
            current_id = [current_patient, current_time]

            if not current_data:
                continue
            else:
                n_segments += current_segments
                data.append(current_data)
                hrs.append(hr)
                labels.append(tag)
                ids.append(current_id)
    return n_segments, data, hrs, labels, ids


def find_largest_segment(data):

    max_len, max_wav_ind, max_seg_ind = 0, 0, 0

    for wav_index in range(len(data)):
        for segment_index in range(len(data[wav_index])):
            current_len = len(data[wav_index][segment_index])
            if current_len > max_len:
                max_len, max_wav_ind, max_seg_ind = current_len, wav_index, segment_index

    return max_len, max_wav_ind, max_seg_ind


def fit_segments(data, max_len, n_segments, hr, label, ids):

    fitted_data, fitted_index = np.zeros((n_segments, max_len)), 0
    fitted_hr, fitted_labels, fitted_padnum = np.zeros(n_segments, dtype=int), np.zeros(n_segments, dtype=int), np.zeros(n_segments, dtype=int)
    fitted_ids = np.zeros((n_segments, 2), dtype=int)

    for wav_index in range(len(data)):
        for segment_index in range(len(data[wav_index])):
            current_seg = np.array(data[wav_index][segment_index])
            pad_num = max_len - current_seg.shape[0]
            current_seg = np.pad(current_seg, (0,pad_num), 'constant', constant_values=(0,0))
            fitted_data[fitted_index, :] = current_seg
            fitted_hr[fitted_index], fitted_labels[fitted_index] = hr[wav_index], label[wav_index]
            fitted_padnum[fitted_index] = pad_num
            fitted_ids[fitted_index, 0], fitted_ids[fitted_index, 1] = ids[wav_index][0], ids[wav_index][1]
            fitted_index += 1

    return fitted_data, fitted_hr, fitted_padnum, fitted_labels, fitted_ids


def create_dataset(data_path, dest_path):

    print("\n#####################\nSaving segments as .npz file: ...\n")
    
    # 1. List all segments:
    n_segments, data, hr, label, ids = list_all_segments(data_path)

    # 2. Find largest valid segment:
    max_len, max_wav_ind, max_seg_ind = find_largest_segment(data)

    # 3. Fit all segments to same length:
    fitted_data, fitted_hr, fitted_padnum, fitted_labels, fitted_ids = fit_segments(data, max_len, n_segments, hr, label, ids)

    # 5. Save in destination path:
    np.savez(dest_path + 'dataset.npz', data=fitted_data, hr=fitted_hr, padnum=fitted_padnum, labels=fitted_labels, ids=fitted_ids)
    
    print("\n Segments saved!")

    return
