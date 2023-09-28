from pre_processing.process_wav import *
from pre_processing.process_csv import *
from utils.utils import plot_n_save
import numpy as np
import os
from tqdm import tqdm


def pre_process_sanollav2(wav_path, ws, target_sr, bp_freq, median_ker, data_path, filename, results_path):
    # pre_process:
    #   This function is a wrapper for all pre-processing methods:
    #       segmentation, normalization, filtration and decimation
    # inputs:
    #   1. wav_path (string) - path to wav_file
    #   2. ws (int) - window size [s]
    #   3. target_sr (int) - target sample rate [Hz]
    #   4. bp_freq (ndarray) - shape (,2) position 1 is pass freq, position 2 is stop freq [Hz]
    #   5. median_ker (int) - size of median_filter kernel
    #   6. csv_arr (ndarray) - values of csv file
    #   7. csv_rows (ndarray) - values of row indices in csv file
    # outputs:
    #   1. windows (ndarray) - normalized valid windows in wav, shape (valid_num, window_samples)
    #   2. sample_rate (int) - achieved sample rate
    windows, init_rate = segment_wav(wav_path, ws, 1)
    # Normalize windows:
    windows = norm_mean(windows)
    # Remove invalid windows:
    windows, num_of_windows, valid_windows = remove_invalid(windows, init_rate)
    # Save raw-data:
    wav_name = wav_path.split('/')[-1]
    if not os.path.exists(data_path + filename):
        os.makedirs(data_path + filename)
    np.save(data_path + filename + '/raw.npy', windows)
    # BandPass, Decimate and Median:
    windows, sample_rate = filter_windows(windows, init_rate, target_sr, bp_freq, median_ker, data_path + filename, results_path)
    # Normalize windows:
    windows = norm_mean(windows)
    # Process peak samples:
    csv_arr, csv_rows, csv_cols = read_csv(wav_path.strip(wav_name) + 'sanollav2_csv.csv')
    peaks, tag = process_peaks(wav_name, valid_windows, init_rate, sample_rate, csv_arr, csv_rows,
                          windows.shape[1], 1)
    # Save Sanolla's estimated HR:
    wav_name_stripped = wav_name.split(".")[0]
    fin_index = wav_name_stripped.rfind('_')
    wav_name_stripped = wav_name_stripped[0:fin_index]
    row_index = np.argwhere(csv_arr == wav_name_stripped)[0][0]
    hr_string = csv_arr[row_index, 5]
    np.save(data_path + filename + '/hrSanolla.npy', int(hr_string))
    np.save(data_path + filename + '/tagSanolla.npy', tag)
    # Add to a single ndarray:
    data = np.stack((windows, peaks), axis=0)
    
    return data, sample_rate , tag

def pre_process_sanollav1(wav_path, ws, target_sr, bp_freq, median_ker, data_path, filename, results_path):
    # pre_process:
    #   This function is a wrapper for all pre-processing methods:
    #       segmentation, normalization, filtration and decimation
    # inputs:
    #   1. wav_path (string) - path to wav_file
    #   2. ws (int) - window size [s]
    #   3. target_sr (int) - target sample rate [Hz]
    #   4. bp_freq (ndarray) - shape (,2) position 1 is pass freq, position 2 is stop freq [Hz]
    #   5. median_ker (int) - size of median_filter kernel
    #   6. csv_arr (ndarray) - values of csv file
    #   7. csv_rows (ndarray) - values of row indices in csv file
    # outputs:
    #   1. windows (ndarray) - normalized valid windows in wav, shape (valid_num, window_samples)
    #   2. sample_rate (int) - achieved sample rate

    prefix = filename[0:2]
    patient_num = filename.split('_')[0]
    wav_name = wav_path.split('/')[-1]
    if prefix=='32':
        diagnosis = find_patient32_diagnosis(patient_num, wav_path.strip(wav_name) + 'patients32.csv')
    elif prefix=='21':
        diagnosis = find_patient21_diagnosis(patient_num, wav_path.strip(wav_name) + 'patients21.csv')
    else:
        return -1, -1, -1

    windows, init_rate = segment_wav(wav_path, ws, 1)
    if windows.size==0 or windows.shape[0] == 1:
        return -1, -1, -1
    # Normalize windows:
    windows = norm_mean(windows)
    # Remove invalid windows:
    windows, num_of_windows, valid_windows = remove_invalid(windows, init_rate)
    # Save raw-data:
    wav_name = wav_path.split('/')[-1]
    if not os.path.exists(data_path + filename):
        os.makedirs(data_path + filename)
    np.save(data_path + filename + '/raw.npy', windows)
    # BandPass, Decimate and Median:
    windows, sample_rate = filter_windows(windows, init_rate, target_sr, bp_freq, median_ker, data_path + filename, results_path)
    # Normalize windows:
    windows = norm_mean(windows)
    # Process peak samples:
    peaks = np.zeros_like(windows)
    # Add to a single ndarray:
    data = np.stack((windows, peaks), axis=0)
    # data = remove_noisy_sections(data, ratio_th, buffer)
    return data, sample_rate, diagnosis

def pre_process_peterj(wav_path, ws, target_sr, bp_freq, median_ker, data_path, filename, results_path):
    # pre_process:
    #   This function is a wrapper for all pre-processing methods:
    #       segmentation, normalization, filtration and decimation
    # inputs:
    #   1. wav_path (string) - path to wav_file
    #   2. ws (int) - window size [s]
    #   3. target_sr (int) - target sample rate [Hz]
    #   4. bp_freq (ndarray) - shape (,2) position 1 is pass freq, position 2 is stop freq [Hz]
    #   5. median_ker (int) - size of median_filter kernel
    #   6. csv_arr (ndarray) - values of csv file
    #   7. csv_rows (ndarray) - values of row indices in csv file
    # outputs:
    #   1. windows (ndarray) - normalized valid windows in wav, shape (valid_num, window_samples)
    #   2. sample_rate (int) - achieved sample rate
 
    s = wav_path.split('/')[-2]
    if s == '1_Normal':
        diagnosis = 0
    elif s == '2_Murmur':
        diagnosis = 1
    else:
        print("Error, ilegal diagnosis folder")

    windows, init_rate = segment_wav(wav_path, ws, 0)
    if windows.size==0 or windows.shape[0] == 1:
        return -1, -1, -1
    # Normalize windows:
    windows = norm_mean(windows)
    # Remove invalid windows:
    windows, num_of_windows, valid_windows = remove_invalid(windows, init_rate)
    # Save raw-data:
    wav_name = wav_path.split('/')[-1]
    if not os.path.exists(data_path + filename):
        os.makedirs(data_path + filename)
    np.save(data_path + filename + '/raw.npy', windows)
    # BandPass, Decimate and Median:
    windows, sample_rate = filter_windows(windows, init_rate, target_sr, bp_freq, median_ker, data_path + filename, results_path)
    # Normalize windows:
    windows = norm_mean(windows)
    # Process peak samples:
    peaks = np.zeros_like(windows)
    # Add to a single ndarray:
    data = np.stack((windows, peaks), axis=0)

    return data, sample_rate, diagnosis

def load_directory_top(raw_path, processed_path, dataset_name, load_parameters, results_path):
    
    if dataset_name == '1_sanollav1/':
        load_sanollav1(raw_path, processed_path, load_parameters, results_path + '1_preprocess/')
    elif dataset_name == '2_sanollav2/':
        load_sanollav2(raw_path, processed_path, load_parameters, results_path + '1_preprocess/')
    elif dataset_name == '3_peterj/':
        load_peterj(raw_path, processed_path, load_parameters, results_path + '1_preprocess/')             
    else:
        print("Dataset name is invalid!")
    
    return


    
    return

def load_sanollav1(raw_path, processed_path, load_parameters, results_path):
    
    ws, target_sr, bp_freq, median_ker, ratio_th, buffer = load_parameters
    
    directory = os.fsencode(raw_path)
    index = 0
    
    print("\n#####################\nPre-processing Sanolla_V1 dataset: \n")

    for file in tqdm(os.listdir(directory), desc='Loading WAVs: ', unit='WAV'):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            
            curr_data, curr_sr, curr_diagnosis = pre_process_sanollav1(raw_path + filename, ws, target_sr, bp_freq,
                                                                    median_ker, processed_path, filename, results_path + str(index))

            if curr_sr==-1:
                continue

            index += 1

            if not os.path.exists(processed_path + filename):
                os.makedirs(processed_path + filename)
            np.save(processed_path + filename + '/data.npy', curr_data)
            np.save(processed_path + filename + '/sr.npy', curr_sr)
            np.save(processed_path + filename + '/diagnosis', curr_diagnosis)

    print("\n Done pre-processing!")
    return

def load_sanollav2(raw_path, processed_path, load_parameters, results_path):
    
    ws, target_sr, bp_freq, median_ker, ratio_th, buffer = load_parameters

    directory = os.fsencode(raw_path)
    index = 0

    print("\n#####################\nPre-processing Sanolla_V2 dataset: \n")

    for file in tqdm(os.listdir(directory), desc='Loading WAVs: ', unit='WAV'):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            
            curr_data, curr_sr, curr_diagnosis = pre_process_sanollav2(raw_path + filename, ws, target_sr, bp_freq,
                                                                    median_ker, processed_path, filename, results_path + str(index))

            if curr_sr==-1:
                continue

            index += 1

            if not os.path.exists(processed_path + filename):
                os.makedirs(processed_path + filename)
            np.save(processed_path + filename + '/data.npy', curr_data)
            np.save(processed_path + filename + '/sr.npy', curr_sr)
            np.save(processed_path + filename + '/diagnosis', curr_diagnosis)

    print("\n Done pre-processing!")
    return

def load_peterj(raw_path, processed_path, load_parameters, results_path):
    
    ws, target_sr, bp_freq, median_ker, ratio_th, buffer = load_parameters
    
    directory_list = [os.fsencode(raw_path + '1_Normal/'), os.fsencode(raw_path + '2_Murmur/')]

    index = 0
    
    for i in range(2):

        print("\n#####################\nPre-processing Peter j bentley dataset, part " + str(i+1) + "/2 : \n")

        for file in tqdm(os.listdir(directory_list[i]), desc='Loading WAVs: ', unit='WAV'):
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                
                if i==0:
                    diag_folder = '1_Normal/'
                else:
                    diag_folder = '2_Murmur/'

                curr_data, curr_sr, curr_diagnosis = pre_process_peterj(raw_path + diag_folder + filename, ws, target_sr, bp_freq,
                                                                        median_ker, processed_path, filename, results_path + str(index))

                if curr_sr==-1:
                    continue

                index += 1

                if not os.path.exists(processed_path + filename):
                    os.makedirs(processed_path + filename)
                np.save(processed_path + filename + '/data.npy', curr_data)
                np.save(processed_path + filename + '/sr.npy', curr_sr)
                np.save(processed_path + filename + '/diagnosis', curr_diagnosis)

    print("\n Done pre-processing!")
    return
