import numpy as np
import scipy.io as sio
import tensorflow as tf
import pdb

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from htnet_model import htnet
from tensorflow.keras import utils as np_utils
# from eeg_emg_load import load_emg_eeg_data, butter_bandpass_filter, epoch, plot_epochs

def fold_split_data(X, y, cur_fold, total_folds):
    # splits the data based on the current fold and returns train and test data
    num_events = X.shape[2]
    split_len = num_events // total_folds

    test_inds = np.arange(0,split_len)+(cur_fold*split_len)
    train_inds = np.setdiff1d(np.arange(num_events),test_inds) #take all events not in test set

    X_train = X[:,:,train_inds]
    y_train = y[train_inds]

    X_test = X[:,:,test_inds]
    y_test = y[test_inds]

    return X_train, y_train, X_test, y_test


def balance_data(grasp_labels, emg_epochs, eeg_epochs):
    rest_indices = np.where(grasp_labels == 4)[0]
    _, num_grasp_events = np.unique(grasp_labels, return_counts = True)
    num_grasp_events = num_grasp_events[0]
    # print(num_grasp_events)
    rand_rest_inds = np.random.choice(rest_indices, size=num_grasp_events, replace=False)
    # print(rand_rest_inds)
    # trimmed_grasp_labels = grasp_labels[rand_rest_inds]
    # np.unique(trimmed_grasp_labels, return_counts = True)

    events_to_keep = np.concatenate( (rand_rest_inds, np.where(grasp_labels == 1)[0], np.where(grasp_labels == 2)[0], np.where(grasp_labels == 3)[0]), axis=0 )
    # print(events_to_keep.shape)
    events_to_keep.sort()
    # print(events_to_keep)

    trimmed_grasp_labels = grasp_labels[events_to_keep]
    print(trimmed_grasp_labels)

    trimmed_emg_epochs = emg_epochs[:,:,events_to_keep]
    print(trimmed_emg_epochs.shape)

    trimmed_eeg_epochs = eeg_epochs[:,:,events_to_keep]
    print(trimmed_emg_epochs.shape)

    print(np.unique(trimmed_grasp_labels, return_counts = True))

    return trimmed_grasp_labels, trimmed_emg_epochs, trimmed_eeg_epochs
    


def train_LDA(folds, cur_channel, epochs, grasp_labels, grasp_names):
    fold_accs = []

    for i in range(folds):
        # need to split the data into folds first
        X_train, y_train, X_test, y_test = fold_split_data(epochs, grasp_labels, i, folds)

        X_train = np.squeeze(X_train[:,cur_channel,:]).T
        X_test = np.squeeze(X_test[:,cur_channel,:]).T

        # then can classify
        lda_classifier = LinearDiscriminantAnalysis()
        lda_classifier.fit(X_train, y_train) #trying with one channel until we get features extracted

        # check predictions from lda
        train_preds = lda_classifier.predict(X_train)
        test_preds = lda_classifier.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        fold_accs.append([train_acc, test_acc])
        print("training accuracy is:", train_acc)
        print("training f1 score is:", f1_score(y_train, train_preds, average='weighted'))
        print("testing accuracy is:", test_acc)
        print("test classification report:")
        print(classification_report(y_test, test_preds, target_names=grasp_names))
        print()

    return fold_accs


def train_RFs(folds, cur_channel, epochs, grasp_labels, grasp_names):
    fold_accs = []

    for i in range(folds):
        # need to split the data into folds first
        X_train, y_train, X_test, y_test = fold_split_data(epochs, grasp_labels, i, folds)

        X_train = np.squeeze(X_train[:,cur_channel,:]).T
        X_test = np.squeeze(X_test[:,cur_channel,:]).T

        # then can classify
        rf_classifier = RandomForestClassifier(max_depth=2)
        rf_classifier.fit(X_train, y_train) #trying with one channel until we get features extracted

        # check predictions from lda
        train_preds = rf_classifier.predict(X_train)
        test_preds = rf_classifier.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        fold_accs.append([train_acc, test_acc])
        print("training accuracy is:", train_acc)
        print("training f1 score is:", f1_score(y_train, train_preds, average='weighted'))
        print("testing accuracy is:", test_acc)
        print("test classification report:")
        print(classification_report(y_test, test_preds, target_names=grasp_names))
        print()

    return fold_accs


def train_SVM(folds, cur_channel, epochs, grasp_labels, grasp_names):
    fold_accs = []

    for i in range(folds):
        # need to split the data into folds first
        X_train, y_train, X_test, y_test = fold_split_data(epochs, grasp_labels, i, folds)

        X_train = np.squeeze(X_train[:,cur_channel,:]).T
        X_test = np.squeeze(X_test[:,cur_channel,:]).T

        # then can classify
        svm_classifier = SVC(gamma='auto')
        svm_classifier.fit(X_train, y_train) #trying with one channel until we get features extracted

        # check predictions from lda
        train_preds = svm_classifier.predict(X_train)
        test_preds = svm_classifier.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        fold_accs.append([train_acc, test_acc])
        print("training accuracy is:", train_acc)
        print("training f1 score is:", f1_score(y_train, train_preds, average='weighted'))
        print("testing accuracy is:", test_acc)
        print("test classification report:")
        print(classification_report(y_test, test_preds, target_names=grasp_names))
        print()

    return fold_accs
    

def train_HTNet(folds, epochs, grasp_labels, grasp_names, fs):
    fold_accs = []

    for i in range(folds):
        # need to split the data into folds first
        X_train, y_train, X_test, y_test = fold_split_data(epochs, grasp_labels, i, folds)

        X_train = np.moveaxis(X_train, -1, 0)
        X_train = np.moveaxis(X_train, -1, -2)
        X_train = np.expand_dims(X_train, axis = 1)
        y_cat_train = np_utils.to_categorical(y_train-1)
        # print(X_train.shape)
        X_test = np.moveaxis(X_test, -1, 0)
        X_test = np.moveaxis(X_test, -1, -2)
        X_test = np.expand_dims(X_test, axis = 1)
        y_cat_test = np_utils.to_categorical(y_test-1)

        # then get model up and running
        # Load NN model
        print("Making model \n")
        # pdb.set_trace()
        htnet_nn = htnet(nb_classes=len(grasp_names), Chans = X_train.shape[2], Samples = X_train.shape[3], data_srate = fs)
        print("Model made\n")

        htnet_opt = tf.keras.optimizers.get("adam")
        htnet_opt.lr.assign(0.001)

        # Set up comiler, checkpointer, and early stopping during model fitting
        htnet_nn.compile(loss="categorical_crossentropy", optimizer=htnet_opt, metrics = ['accuracy'])

#         pdb.set_trace()
        htnet_nn.summary()
        train_preds = htnet_nn.predict(X_train).argmax(axis = -1)
        train_acc = accuracy_score(y_train, train_preds)
        print(train_preds)
        print("training accuracy before training is:", train_acc)
        
        # Perform model fitting in Keras
        htnet_nn.fit(X_train, y_cat_train, batch_size = 16, epochs = 50, verbose = 2)
        
        htnet_nn.evaluate(X_train, y_cat_train)
        htnet_nn.evaluate(X_test, y_cat_test)
        
        # get the accuracies
        train_preds = (htnet_nn.predict(X_train).argmax(axis = -1)) + 1
        test_preds = (htnet_nn.predict(X_test).argmax(axis = -1)) + 1
        from sklearn.metrics import log_loss
#         pdb.set_trace()
        print(log_loss(y_train, htnet_nn.predict(X_train), labels=[1,2,3,4] ) )
        print(train_preds.shape)
        print(train_preds)
        print(y_train.shape)
        print(y_train)
        print()
        print(test_preds.shape)
        print(test_preds)
        
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        fold_accs.append([train_acc, test_acc])
        print("training accuracy is:", train_acc)
        print("training f1 score is:", f1_score(y_train, train_preds, average='weighted'))
        print("testing accuracy is:", test_acc)
        print("test classification report:")
        print(classification_report(y_test, test_preds, target_names=grasp_names, labels=range(len(grasp_names))) )
        print()
        
        tf.keras.backend.clear_session() # avoids slowdowns when running fits for many folds

    return fold_accs