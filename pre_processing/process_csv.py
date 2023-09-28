import numpy as np
import pandas as pd
import os


def find_patient32_diagnosis(patient_num, csv_path):
    patients_data = pd.read_csv(csv_path).to_numpy(dtype=str)
    row_idx = np.argwhere(patients_data[:,0]==patient_num)
    if row_idx.size==0:
        return -1
    else:
        diagnosis = patients_data[row_idx, 2][0][0]
        if diagnosis == 'nan':
            return 0
        return 1

def find_patient21_diagnosis(patient_num, csv_path):
    patients_data = pd.read_csv(csv_path).to_numpy(dtype=str)
    patients_data = pd.read_csv(csv_path).to_numpy(dtype=str)
    row_idx = np.argwhere(patients_data[:,0]==patient_num)
    consider_ill = ['Moderate', 'moderate', 'severe', 'Severe', 'Moderate-Severe']
    if row_idx.size==0:
        return -1
    else:
        ill = 0
        current_deg = np.squeeze(patients_data[row_idx, 3:10:2])
        for illness_deg in consider_ill:
            if illness_deg in current_deg:
                ill = 1
        return ill

def read_csv(path):
    data = pd.read_csv(path, header=0, index_col=0)
    array = data.values
    rows = data.index.values
    cols = data.columns.values
    return array, rows, cols


def get_samples(wav_name, initial_rate, final_rate, array, rows):
    row_index = np.argwhere(rows == wav_name)
    first = False
    if np.isnan(array[row_index, 0]):
        first = True
    samples = array[row_index, :]
    samples = samples[~np.isnan(samples)]
    factor = final_rate / initial_rate
    samples = (samples*factor).astype(np.uint)
    return samples, first


def get_samples_sanolla(wav_name, sr, array):
    wav_name_stripped = wav_name.split(".")[0]
    fin_index = wav_name_stripped.rfind('_')
    wav_name_stripped = wav_name_stripped[0:fin_index]
    row_index = np.argwhere(array == wav_name_stripped)[0][0]
    seconds_string = array[row_index, 6]
    tag = array[row_index, 3]
    if seconds_string != seconds_string:
        return -1, -1
    else:
        seconds_string = seconds_string.replace(' ', '')
        seconds = seconds_string.split(',')
        seconds = np.asarray(seconds, float)
        seconds -= 1  # Removed first second of wav in all sanolla data
        samples = (seconds * sr).astype(int)
        return samples, tag


def divide_peaks(peak_samples, first_stage):
    s1_peaks = peak_samples[first_stage:-1:2]
    first_stage = not first_stage
    s2_peaks = peak_samples[first_stage:-1:2]
    return s1_peaks, s2_peaks


def segment_peaks(window, peaks, ws):
    lower_bound = int(ws * window)
    upper_bound = lower_bound + ws - 1
    condition = (peaks >= lower_bound) & (peaks <= upper_bound)
    curr = np.where(condition, peaks, 0)
    curr = curr[curr > 0]
    curr -= lower_bound
    return curr


def standardize_peaks(s1peaks, s2peaks, ws):
    stand = np.zeros(ws)
    stand[s1peaks] = 1
    stand[s2peaks] = 2
    return stand


def process_peaks(wav_name, valid_windows, init_rate, sample_rate, csv_arr, csv_rows, ws, sanolla):
    num_of_windows = int(valid_windows.shape[0])
    if sanolla:
        s1_peaks, tag = get_samples_sanolla(wav_name, sample_rate, csv_arr)
    else:
        peak_samples, first_stage = get_samples(wav_name, init_rate, sample_rate, csv_arr, csv_rows)
        s1_peaks, s2_peaks = divide_peaks(peak_samples, first_stage)
        tag = -1
    peaks = np.zeros((num_of_windows, ws))
    index = 0
    for window in valid_windows:
        curr_s1 = segment_peaks(window, s1_peaks, ws)
        if sanolla:
            peaks[index, curr_s1] = 1
        else:
            curr_s2 = segment_peaks(window, s2_peaks, ws)
            curr_peaks = standardize_peaks(curr_s1, curr_s2, ws)
            peaks[index, :] = curr_peaks
        index += 1
    return peaks, tag