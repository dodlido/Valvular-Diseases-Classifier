import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import tree
from numpy import genfromtxt
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv


def classify_svm(dataset_path):

    print("\n#####################\nSVM Classification... \n")
    
    # Read dateset:
    dataset = genfromtxt(dataset_path + 'features.csv', delimiter=',', skip_header=1)
    labels = dataset[1:, -1]
    indices = dataset[1:, 0]
    features = dataset[1:, 1:-1]

    ok_indices = np.argwhere(labels!=-1)
    labels = np.squeeze(labels[ok_indices])
    indices = np.squeeze(indices[ok_indices])
    features = np.squeeze(features[ok_indices, :])

    X_train, X_test, y_train, y_test = \
        train_test_split(features, labels, test_size=0.2, random_state=109)

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets:
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred)
    print("Done classification process! \nAccuracy: ", accuracy * 100, "%")

    return accuracy*100, f1*100


def classify_tree(dataset_path, result_path):

    print("\n#####################\nTree Classification... \n")
    
    dataset = genfromtxt(dataset_path + 'melvectors.csv', delimiter=',', skip_header=1)
    cycle_index = dataset[:, 0]
    patient_id = dataset[:, 1]
    recording_time = dataset[:, 2]
    heartrate = dataset[:, 3]
    padnum = dataset[:, 4]
    labels = dataset[:, 5]
    features = dataset[:, 6:]

    ok_indices = np.argwhere(labels!=-1)
    labels = np.squeeze(labels[ok_indices])
    features = np.squeeze(features[ok_indices, :])

    test_size = int(0.2 * labels.shape[0])
    test_indices = np.random.randint(low=labels.shape[0], size=test_size)   
    X_train, X_test, y_train, y_test = features[~test_indices], features[test_indices], labels[~test_indices], labels[test_indices]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    fig = plt.figure(figsize=(12,12))
    tree.plot_tree(clf)
    fig.tight_layout()
    fig.savefig(result_path + '/dt_vis.png')

    y_pred = clf.predict(X_test)

    apr_pred = np.zeros_like(y_pred)

    for i in range(apr_pred.shape[0]):
        
        same_id = np.squeeze(np.argwhere(patient_id[test_indices]==patient_id[test_indices[i]]))
        same_time = np.squeeze(np.argwhere(recording_time[test_indices]==recording_time[test_indices[i]]))
        same_recording = np.intersect1d(same_id, same_time)

        num_sick = np.sum(y_pred[same_recording])

        if num_sick > same_recording.shape[0] // 2:
            apr_pred[i] = 1


    # Model Accuracy: how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    apr_accuracy = metrics.accuracy_score(y_test, apr_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    print("Done classification process! \nAccuracy: ", accuracy * 100, "%")

    return accuracy*100, f1*100, apr_accuracy*100


def classify_multiple_svms(dataset_path):

    print("\n#####################\nMultiple models classification... \n")

    # Read dateset:
    dataset = genfromtxt(dataset_path + 'features.csv', delimiter=',', skip_header=1)
    labels = dataset[1:, -1]
    dataset_indices = dataset[1:, 0]
    cycles_indices = dataset[1:, 1]
    features = dataset[1:, 2:-1]

    ok_indices = np.argwhere(labels!=-1)
    labels = np.squeeze(labels[ok_indices])
    indices = np.squeeze(indices[ok_indices])
    features = np.squeeze(features[ok_indices, :])

    indices_sick = np.argwhere(labels==1)
    labels_sick = labels[indices_sick]
    features_sick = features[indices_sick, :]
    X_train_sick, X_test_sick, y_train_sick, y_test_sick = \
        train_test_split(features_sick, labels_sick, test_size=0.2, random_state=109)

    indices_healthy = np.argwhere(labels==0)
    labels_healthy = labels[indices_healthy]
    features_healthy = features[indices_healthy, :]
    X_train_healthy, X_test_healthy, y_train_healthy, y_test_healthy = \
        train_test_split(features_healthy, labels_healthy, test_size=0.2, random_state=109)

    X_test = np.squeeze(np.concatenate((X_test_sick, X_test_healthy), axis=0))
    y_test = np.squeeze(np.concatenate((y_test_sick, y_test_healthy), axis=0))

    sick_train_num, health_train_num = y_train_sick.shape[0], y_train_healthy.shape[0]
    classifiers_num = 7

    predictions_array = np.zeros((classifiers_num, y_test.shape[0]))

    for i in range(classifiers_num):

        healthy_indices = np.random.randint(low=health_train_num, size=4*sick_train_num)

        current_X_train = np.squeeze(np.concatenate
                                     ((X_train_sick, X_train_healthy[healthy_indices, :]), axis=0))
        current_y_train = np.squeeze(np.concatenate
                                     ((y_train_sick, y_train_healthy[healthy_indices, :]), axis=0))

        # Create a svm Classifier
        clf = svm.SVC(kernel='linear')  # Linear Kernel

        # Train the model using the training sets:
        clf.fit(current_X_train, current_y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)
        predictions_array[i, :] = y_pred

        # Model Accuracy: how often is the classifier correct?
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy of model ", str(i+1), "/", str(classifiers_num), " : ", accuracy * 100, "%")

    final_prediction = np.sum(predictions_array, axis=0)
    final_prediction = np.where(final_prediction > classifiers_num // 2, 1, 0)
    accuracy = metrics.accuracy_score(y_test, final_prediction)
    confusion = metrics.confusion_matrix(y_test, final_prediction)
    f1 = metrics.f1_score(y_test, y_pred)
    print("\nDone classification process! \nAccuracy: ", accuracy * 100, "%")
    # print("\n Confusion Matrix: \n", confusion)

    return accuracy*100, f1*100


def classifier_top(path_features, path_results, paths_featype, dataset_name):
       
    
    # if dataset_name=='2_sanollv2':
    #     accuracy_svm, f1_svm = classify_multiple_svms(path_features + paths_featype[0])
    
    # else:
    #     accuracy_svm, f1_svm = classify_svm(path_features + paths_featype[0])
    accuracy_svm, f1_svm = 0, 0
    
    accuracy_tree, f1_tree, apr_accuracy = classify_tree(path_features + paths_featype[2], path_results)

    with open(path_results + 'accuracy.txt', 'a') as f:
        f.write('Dataset ' + dataset_name + ' SVM classification accuracy score is: ' + str(accuracy_svm) + '%\n')
        f.write('Dataset ' + dataset_name + ' SVM classification f1 score is: ' + str(f1_svm) + '%\n')
        f.write('Dataset ' + dataset_name + ' DT classification accuracy score is: ' + str(accuracy_tree) + '%\n')
        f.write('Dataset ' + dataset_name + ' DT classification f1 score is: ' + str(f1_tree) + '%\n')
        f.write('Dataset ' + dataset_name + ' DT classification accuracy per recording score is: ' + str(apr_accuracy) + '%\n')
    
    return accuracy_tree, accuracy_svm, apr_accuracy


def classifier_combined(root_dir, path_feats, path_dataset_list, path_featype_list, path_results):

    path_svm_feats1 = root_dir + path_feats + path_dataset_list[0] + path_featype_list[0]
    path_svm_feats2 = root_dir + path_feats + path_dataset_list[1] + path_featype_list[0]
    path_svm_feats3 = root_dir + path_feats + path_dataset_list[2] + path_featype_list[0]

    svm_feats1 = genfromtxt(path_svm_feats1 + 'features.csv', delimiter=',', skip_header=1)
    svm_feats2 = genfromtxt(path_svm_feats2 + 'features.csv', delimiter=',', skip_header=1)
    svm_feats3 = genfromtxt(path_svm_feats3 + 'features.csv', delimiter=',', skip_header=1)

    svm_featstot = np.append(svm_feats1, svm_feats2, axis=0)
    svm_featstot = np.append(svm_featstot, svm_feats3, axis=0)

    combined_svm_path = root_dir + path_feats + path_dataset_list[3] + path_featype_list[0]
    combine_features_as_csv(svm_featstot, combined_svm_path)
    combined_svm_acc, combined_svm_f1 =  classify_svm(combined_svm_path)

    path_dt_feats1 = root_dir + path_feats + path_dataset_list[0] + path_featype_list[2]
    path_dt_feats2 = root_dir + path_feats + path_dataset_list[1] + path_featype_list[2]
    path_dt_feats3 = root_dir + path_feats + path_dataset_list[2] + path_featype_list[2]
    
    dt_feats1 = genfromtxt(path_dt_feats1 + 'melvectors.csv', delimiter=',', skip_header=1)
    dt_feats2 = genfromtxt(path_dt_feats2 + 'melvectors.csv', delimiter=',', skip_header=1)
    dt_feats3 = genfromtxt(path_dt_feats3 + 'melvectors.csv', delimiter=',', skip_header=1)

    dt_featstot = np.append(dt_feats1, dt_feats2, axis=0)
    dt_featstot = np.append(dt_featstot, dt_feats3, axis=0)

    combined_dt_path = root_dir + path_feats + path_dataset_list[3] + path_featype_list[2]
    combine_melv_as_csv(dt_featstot, combined_dt_path)
    combined_dt_acc, combined_dt_f1 = classify_tree(combined_dt_path)

    with open(root_dir + path_results + path_dataset_list[3] + 'accuracy.txt', 'a') as f:
        f.write('Combined Dataset SVM classification accuracy score is: ' + str(combined_svm_acc) + '%\n')
        f.write('Combined Dataset SVM classification F1 score is: ' + str(combined_svm_f1) + '%\n')
        f.write('Combined Dataset DT classification accuracy score is: ' + str(combined_dt_acc) + '%\n')
        f.write('Combined Dataset SVM classification F1 score is: ' + str(combined_dt_f1) + '%\n')

    return

def combine_features_as_csv(array, path):
    
    with open(path + 'features.csv', 'w', newline='') as file:
        
        writer = csv.writer(file)
        field = ["cycle_#", "zero_cross", "mean_mfcc", "std_mfcc", "spectral_centroid_1",
                 "spectral_centroid_2", "spectral_centroid_3", "spectral_rolloff_1",
                 "spectral_rolloff_2", "spectral_rolloff_3", "spectral_flux", "mean_frequency_real",
                 "mean_frequency_imaginary", "energy_entropy", "pad_num", "label"]
        writer.writerow(field)

        for i in tqdm(range(array.shape[0]), desc='writing csv', unit='cycle'):
            
            row = array[i]
            writer.writerow(row)
    
    print("\n Writing combined csv done!")
    
    return

def combine_melv_as_csv(array, path):
    
    with open(path + 'melvectors.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["cycle_#", "patient_id", "time", "hr", "padnum", "label"]
        for i in range(array.shape[1] - len(field)):
            field.append("s_"+str(i))
        writer.writerow(field)

        for i in tqdm(range(array.shape[0]), desc='writing csv', unit='cycle'):
    
            row = array[i]
            writer.writerow(row)          

    print("\n Writing combined csv done!")

    return
 