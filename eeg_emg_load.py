# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 07:55:20 2021

@author: miaea
"""

def load_emg_eeg_data(filename):
    """
    

    Parameters
    ----------
    filename : str
        Complete filename with path for EMG or EEG data.

    Returns
    -------
    channels : (Number of samples, Number of channels) array
        Array containing the data of every channel.
        Ex (37883352, 7)
    fs : Integer
        Sampling frequency of the signal
        Ex/ 2500 Hz
    event_times : (Number of events,) array
        The timings of the event showings.
        Ex/ event_times[0] = 15455
    event_type : (Number of events,) array
        The event type specified as an integer at the timings given by event_times.
        Ex/ event_type[0] = 1
    event_type_labels : (Number of types, ) array
        Array containing the text labels to map from the event_type array
        to the goal condition. Ex/ event_type_labels[0] = 'Cylindrical' 

    """
    import numpy as np
    import scipy.io as sio
    
    data = sio.loadmat(filename)
    
    # First let's assemble all of the channels into one array
    all_variables = getKeys(data)
    for name in all_variables:
        if name.startswith('ch'): 
            if ('channels' in locals()) == 0:
                channels = data[name]; 
            else:
                channels = np.append(channels, data[name],1)
    
    # Now we should also get the event data and timing into an array
    event_times = data['mrk']['pos'][0][0][0]
    fs = data['mrk']['fs'][0][0][0][0]
    
    y = data['mrk']['y'][0][0]
    event_type = np.ones(np.shape(y)[1])
    for i in range(np.shape(y)[0]):
        event_type[y[i,:] == 1] = i+1;
    event_type_labels = data['mrk']['className'][0][0][0];
    
    # Let's also get starting and stopping times
    t_start = data['mrk']['misc'][0][0][0]['pos'][0][0][0];
    t_stop = data['mrk']['misc'][0][0][0]['pos'][0][0][1];
    
    
    # Let's now clean the data so that we only have the times 
    #where data was actively running
    channels = channels[t_start-1:t_stop-1,:] # remove channels of non-interest
    event_times = event_times - t_start # shift event times
    
    return channels, fs, event_times, event_type, event_type_labels

def getKeys(dict):
    """ Get the headers of a dictionary"""
    list = [];
    for key in dict.keys():
        list.append(key)
    return list

def epoch(channels, event_times, fs,seconds):
    """
    

    Parameters
    ----------
    channels : [Number of samples, Number of channels] array
        The EMG or EEG data that is to be broken up into epochs.
    event_times : [Number of events,] array
        The timings of the events. These are where the epochs will be split up.
    fs : Integer.
        The sampling frequency for the signal.
    seconds : Integer.
        The desired duration of each epoch in seconds.

    Returns
    -------
    epoched : [Seconds*fs,Number of channels, Number of events] array
        The signal data for each channel broken into epochs of the desired length.

    """
    import numpy as np
    
    # Now we are going to break the signal up into X seconds epochs
    epoched = np.zeros([fs*seconds, channels.shape[1], event_times.size])
    for et_indx in range(event_times.size):
            epoched[:,:, et_indx] = channels[event_times[et_indx]:event_times[et_indx]+(seconds*fs),:]
    return epoched


def plot_epochs(data, fs, N_channels, N_epochs):
    """
    

    Parameters
    ----------
    data :  [Seconds*fs,Number of channels, Number of events] array
        The signal data for each channel broken into epochs of the desired length.
    fs : integer
        The sampling frequency.
    N_channels : Integer
        Number of channels to plot. [Plots 0:N_channels]
    N_epochs : Integer
        Number of Epochs to plot. [Plots 0:N_epochs]

    Returns
    -------
    None.

    """

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    for i in range(N_epochs):
        plt.plot(np.linspace(0, data.shape[0]/fs, data.shape[0]),
             data[:,0:N_channels, i])
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.xlim([0, data.shape[0]/fs])
    
    def butter_bandpass(lowcut, highcut, fs, order=5):
    "https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html"
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    "https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html"
    from scipy.signal import butter, lfilter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


""" 
    Main Code Here

"""
import os

" Set file names"
# Set path to file
path = 'G:/.shortcut-targets-by-id/16bjLJ-x3DCMytjvubbbvXEirA9t2l2q-/FinalProjectData';

eeg_folder = 'EEG'
eeg_file = 'EEG_session1_sub1_multigrasp_realMove.mat';

emg_folder = 'EMG_ConvertedData';
emg_file = 'EMG_session1_sub1_multigrasp_realMove.mat';

" Load file"
emg_data, emg_fs, emg_event_times, emg_grasp_labels, emg_grasp_names = load_emg_eeg_data(os.path.join(path,emg_folder, emg_file));

eeg_data, eeg_fs, eeg_event_times, eeg_grasp_labels, eeg_grasp_names = load_emg_eeg_data(os.path.join(path,eeg_folder, eeg_file));

" Apply Butter Bandpass Filter"
emg_data = butter_bandpass_filter(emg_data, 8, 30, emg_fs, order=5)
eeg_data = butter_bandpass_filter(eeg_data, 8, 30, emg_fs, order=5)

" Break into epochs"

emg_epochs = epoch(emg_data,emg_event_times, emg_fs, 1);
eeg_epochs = epoch(eeg_data,eeg_event_times, eeg_fs, 1);

" Plot epochs for visualization "

plot_epochs(emg_epochs, emg_fs, 1, 10) # First number is number of channels, Second number is number of epochs
plot_epochs(eeg_epochs, eeg_fs, 1, 10) # First number is number of channels, Second number is number of epochs

" Now let's see if we can plot the subset for a specified label"
plot_epochs(emg_epochs[:,:,emg_grasp_labels == 1], emg_fs, 1, 10)
