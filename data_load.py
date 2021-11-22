import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_vars(filename):
    data = sio.loadmat(filename)
    
    # get the y labels
    grasp_labels = data['mrk']['y'][0][0]
    grasp_labels = grasp_labels.T
    print(grasp_labels)

    # get the grasp names
    tmp_grasp_names = data['mrk']['className'][0][0][0]
    grasp_names = []
    for n in tmp_grasp_names:
        grasp_names.append(n[0])
    print(grasp_names)

    # get the event times
    event_times = data['mrk']['pos'][0][0][0]
    print(event_times)

    # get the sampling rate (should be the same across all data)
    eeg_fs = data['mrk']['fs'][0][0][0][0]
    print(eeg_fs)

    # get the number of channels
    num_chans = data['mnt']['x'][0][0].shape[0]
    print(num_chans)

    # get the total number of eeg samples
    time_samples = data['ch1'].shape[0]
    print(time_samples)

    # get the actual eeg data out
    eeg_data = np.zeros( (num_chans, time_samples) )
    for chan in range(num_chans):
        eeg_data[chan] = np.squeeze(data['ch'+str(chan+1)])

    print(eeg_data.shape)

    return grasp_labels, grasp_names, event_times, eeg_fs, num_chans, eeg_data

def load_and_epoch(data_loc, data_filenames, seconds):
    # load in for defining size vars
    grasp_labels, grasp_names, event_times, fs, num_chans, data = load_data_vars(data_loc+data_filenames[0])
    epoched_eeg = np.zeros( (len(data_filenames)*len(event_times), num_chans, seconds*eeg_fs) )
    y_labels = np.zeros( (len(data_filenames)*len(event_times), len(grasp_labels.shape[0])) )
    for file_num, fn in enumerate(data_filenames):
        # get all the relevant variables
        grasp_labels, grasp_names, event_times, fs, num_chans, data = load_data_vars(data_loc+fn)
        
        # create the data epochs
        for i, et in enumerate(event_times):
            y_labels[(file_num*len(event_times)) + i,:] = grasp_labels[i]
            epoched_eeg[(file_num*len(event_times)) + i,:,:] = data[:,et:et+(1*fs)]

    return epoched_eeg, y_labels




data_loc = "data/EEG/"
data_filenames = ["EEG_session1_sub1_multigrasp_realMove.mat", 
                    "EEG_session2_sub1_multigrasp_realMove.mat", 
                    "EEG_session3_sub1_multigrasp_realMove.mat"]

seconds = 1

epoched_eeg, y_labels = load_and_epoch(data_loc, data_filenames, seconds)