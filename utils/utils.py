import numpy as np
import os
from matplotlib import pyplot as plt

def unpack_paths_txt(paths_txt):
    
    lines = []

    with open(paths_txt) as file:
        lines.append(file.readlines())
    
    for index in range(len(lines[0])):
        lines[0][index] = lines[0][index].strip('\n')
    
    paths_stage = np.array([lines[0][i] for i in np.arange(start=1, stop=6)])
    paths_dataset = np.array([lines[0][i] for i in np.arange(start=7, stop=11)])
    paths_featype = np.array([lines[0][i] for i in np.arange(start=12, stop=len(lines[0]))])

    return paths_stage, paths_dataset, paths_featype

def unpack_parameters(params_txt):
    
    lines = []

    with open(params_txt) as file:
        lines.append(file.readlines())
    
    for index in range(len(lines[0])):
        lines[0][index] = lines[0][index].strip('\n')
    
    window_size = int(lines[0][0])
    target_sr = int(lines[0][1])
    bandpass_f = [int(lines[0][2]), int(lines[0][3])]
    median_ker = int(lines[0][4])
    noise_ratio_th = int(lines[0][5])
    ratio_buff = int(lines[0][7])

    load_params = [window_size, target_sr, bandpass_f, median_ker, noise_ratio_th, ratio_buff]
    
    return load_params

def unpack_stage(root_dir, paths_dataset, paths_stage, data_ind):

    dataset_name = paths_dataset[data_ind]
    current_stage = [root_dir + s + dataset_name for s in paths_stage]

    return dataset_name, current_stage

def plot_n_save(sig, sr, title, path, s1_locs=None):
    sig = np.squeeze(np.reshape(sig, -1))
    length = sig.shape[0]
    time_axis = np.linspace(start=0, stop=length/sr, num=length)
    fig, ax = plt.subplots(1,1)
    ax.plot(time_axis, sig)
    if s1_locs is not None:
        s1_locs = (s1_locs / sr).astype('int')
        for i, loc in enumerate(s1_locs):
            x, y = np.array([loc, loc]), np.array([sig.min(), sig.max()])
            if i==0:
                ax.plot(x, y, 'g--', label='S1')
            else:
                ax.plot(x, y, 'g--')
        ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amp [A.U]')
    ax.set_title(title)
    fig.tight_layout()
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(path + '/' + title + '.png')
    return

def visualize_single_feature(feature_name, feature_data, labels, img_path):

    sick = feature_data[np.argwhere(labels==1)]
    sicky = 1 * np.ones_like(sick)
    healthy = feature_data[np.argwhere(labels==0)]
    healthyy = -1 * np.ones_like(healthy)

    plt.scatter(sick, sicky, c='#1f77b4', marker='o', s=12, label='Sick')
    plt.scatter(healthy, healthyy, c='#ff7f0e', marker='o', s=12, label='Healthy')
    plt.title(feature_name)
    plt.xlabel('Feature Distibution')
    plt.legend()
    plt.savefig(img_path + feature_name + '.png')
    plt.clf()

    return

def visualize_all_features_1d(features_names, features_path, img_path):

    dataset = np.genfromtxt(features_path, delimiter=',', skip_header=1)

    labels = dataset[:, -1]
    features = dataset[:, 1:-1] 

    for i in range(len(features_names)):  

        visualize_single_feature(features_names[i], features[:, i], labels, img_path)

    return

def save_spectogram(spectogram, save_path):

    fig, ax = plt.subplots(1, 1)
    
    # Mel Spectogram:
    ax.imshow(spectogram)
    ax.set_title('Spect')
    ax.set_axis_off()

    fig.tight_layout()

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fig.savefig(save_path + 'spect.png')

    return
