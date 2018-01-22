
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
from tqdm import tqdm
# Math
import numpy as np
import scipy.stats
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
from scipy import sparse, stats, spatial
import scipy.sparse.linalg

# Machine learning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import  confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd

# Cutting

from cut_audio import *




def main_train_audio_extraction():
    '''
    - Function that allow the extraction of all the audio files.
    - Process :
    1. Indexing the path, the class and the speaker of all the audio files.
    2. Audio Extraction :
        2.1. Loading all the audio files into memory
        2.2. Detecting the position of the word inside each audio files and cutting them
        2.3. Saving into a Pickled DataFrame all the audio and their cutted version
    '''

    train_audio_path = join('..','Project','data','train','audio')

    # Listing the directories of each word class
    dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
    dirs.sort()

    path = []
    word = []
    speaker = []
    iteration = []

    # Loading the information of the audio files
    for direct in dirs:
        if not direct.startswith('_'):

            list_files = os.listdir(join(train_audio_path, direct))
            wave_selected  = list([ f for f in list_files if f.endswith('.wav')])

            # Extraction of file informations for dataframe
            word.extend(list(np.repeat(direct,len(wave_selected),axis=0)))
            speaker.extend([wave_selected[f].split('.')[0].split('_')[0] for f in range(len(wave_selected)) ])
            iteration.extend([wave_selected[f].split('.')[0].split('_')[-1] for f in range(len(wave_selected)) ])
            path.extend([train_audio_path + '/' + direct + '/' + wave_selected[f] for f in range(len(wave_selected))])

    # Saving those informations into a pandas DataFrame
    features_og = pd.DataFrame({('info','word',''): word,
                                ('info','speaker',''): speaker,
                                ('info','iteration',''): iteration,
                                ('info','path',''): path})
    index_og = [('info','word',''),('info','speaker',''),('info','iteration','')]

    print('Number of signals : ' + str(len(features_og)))


    # Load and cut the audio files.
    raw_audio_df = load_audio_file(features_og)

    # Save the raw audio Dataframe into a set a pickles :
    i = 0
    k = 0
    while True :
        i_next = i + 6000
        k += 1
        if i_next < len(raw_audio_df) :
            raw_audio_df.iloc[i:i_next].to_pickle(('../Project/data/raw_audio_all_'+ str(k)+'.pickle'))
        else :
            raw_audio_df.iloc[i:len(raw_audio_df)].to_pickle(('../Project/data/raw_audio_all_'+ str(k)+'.pickle'))
            break

        i = i_next

def main_train_audio_features():
    '''
    - Function that allow the computation of all the features.
    - Process :
    1. Load the raw audio files.
    2. Features Extraction :
        2.1. Loading the Previously pickled raw audio file.
        2.2. Computing the MFCC of all the cutted version of the audio files.
        2.3. Saving them in a Pickled Pandas DataFrame
    '''
    audio_loaded_df = pd.read_pickle(('../Project/data/raw_audio_all_'+ str(1)+'.pickle'))

    for i in range(2,12):
        audio_loaded_df = audio_loaded_df.append(pd.read_pickle(('../Project/data/raw_audio_all_'+ str(i)+'.pickle')))

    # Optimal Parameters :
    N_MFCC = 10
    N_FFT =  int(2048/2)
    NUM_MFCCS_VEC = 20
    audio_loaded_df = audio_loaded_df.drop(2113).reset_index(drop=True)
    features_og = compute_mfcc_raw(audio_loaded_df,N_MFCC,N_FFT,NUM_MFCCS_VEC,cut=True)

    # Save features DataFrame as pickle
    features_og.drop(axis=1,columns=('audio')).to_pickle('./Features Data/cut_mfccs_all_raw_10_1028_20.pickle')
    features_og.head(2)


def load_audio_file(features_og):

    print("----- Start Importation -----")

    count_drop = 0
    audio_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('audio','raw',''),('audio','sr',''),('audio','cut','')]),index=features_og.index)

    for w in tqdm(range(len(features_og)),total=len(features_og),unit='waves'):

        audio, sampling_rate = librosa.load(features_og[('info','path')].iloc[w], sr=None, mono=True)

        clean_condition = (np.max(audio) != 0.0)

        if clean_condition:
            audio_df.loc[w,('audio','raw','')] = audio
            audio_df.loc[w,('audio','sr','')] = sampling_rate
            audio_df.loc[w,('audio','cut','')] = cut_signal(audio)
        else :
            count_drop += 1
            audio_df.drop(w)
            features_og.drop(w)


    audio_df = features_og.merge(audio_df,left_index=True,right_index=True)
    print("----- End Importation -----")
    print("Number of dropped signals :",count_drop)
    return audio_df


def compute_mfcc_raw(features_og,N_MFCC,N_FFT,NUM_MFCCS_VEC,cut=True):
    '''
    This function computes the raw MFCC parameters for and allow the choice of parameters
    '''

    stat_name= ['raw_mfcc']
    col_names = [('mfcc',stat_name[i],j) for i in range(len(stat_name))  for j in range(N_MFCC*NUM_MFCCS_VEC)]

    features_mfcc = pd.DataFrame(columns=pd.MultiIndex.from_tuples(col_names),index=features_og.index)
    # sorting the columns in order to improve index performances (see lexsort errors)
    features_mfcc.sort_index(axis=1,inplace=True,sort_remaining=True)

    # MFCC FEATURES :
    for w in tqdm(range(len(features_og)),total=len(features_og),unit='waves'):
        # Handling the cut version of the signal :
        if cut == True :
            audio = features_og.loc[w,('audio','cut','')]
        else :
            audio = features_og.loc[w,('audio','raw','')]

        sampling_rate = features_og.loc[w,('audio','sr','')]

        # Computing the MFCC for each signal :
        mfcc = librosa.feature.mfcc(y=audio,sr=sampling_rate, n_mfcc=N_MFCC, n_fft = N_FFT, hop_length = int(np.floor(len(audio)/NUM_MFCCS_VEC)))

        features_mfcc.loc[w, ('mfcc', 'raw_mfcc')] = mfcc[:,:-1].reshape(-1,)

    features_og = features_og.merge(features_mfcc,left_index=True,right_index=True)
    return features_og

# pipeline function for semisupervised learning using graphs
def semisup_test_all_dataset(features_og, label_vec, train_x, train_y, test_x, batch_size, NEIGHBORS, alpha, beta):
    """Test semisupervised graph learning algorithm for entire dataset.
    - features_og : original copy of all MFCCs
    - train_x : indices of training samples in dataset
    - train_y : labels of training samples
    - test_x : indices of testing samples in dataset
    - batch_size : number of samples to be predict per iteration in main loop
    - NEIGHBORS : number of neirest neighbors in k-NN sparsification
    - alpha : hyper-parameter which controls the trade-off between the data fidelity term and the smoothness prio
    - beta :
    """

    # number of batches to loop through
    n_batch = int(len(test_x) / batch_size)

    # encode training samples classes into 1-hot array
    n_class = np.max(train_y)
    Y = np.eye(len(class_names))[train_y - 1].T

    # accumulate accuracy values
    accuracy_tot = []
    accuracy = []
    remaining_test = np.array(test_x)

    for batch in tqdm(range(n_batch)):

        # get batch indices
        potential_elements  = np.array(list(enumerate(remaining_test)))
        indices = np.random.choice(potential_elements[:,0].reshape(-1,), batch_size, replace=False)
        batch_index = potential_elements[:,0].reshape(-1,)[indices]
        remaining_test = np.delete(remaining_test, indices)

        # build graph
        features = pd.DataFrame(features_og['mfcc'], np.append(train_x, batch_index))
        features -= features.mean(axis=0)
        features /= features.std(axis=0)

        distances = spatial.distance.squareform(spatial.distance.pdist(features,'cosine'))

        n=distances.shape[0]
        kernel_width = distances.mean()
        weights = np.exp(np.divide(-np.square(distances),kernel_width**2))
        np.fill_diagonal(weights,0)

        # k-NN sparsification
        for i in range(weights.shape[0]):
            idx = weights[i,:].argsort()[:-NEIGHBORS]
            weights[i,idx] = 0
            weights[idx,i] = 0

            # compute laplacian
            degrees = np.sum(weights,axis=0)
            laplacian = np.diag(degrees**-0.5) @ (np.diag(degrees) - weights) @ np.diag(degrees**-0.5)
            laplacian = sparse.csr_matrix(laplacian)

            # add test samples to 1-hot array
            M = np.zeros((len(class_names), len(train_y) + batch_size)) # mask matrix
            M[:len(train_y),:len(train_y)] = 1
            Y_compr = np.concatenate((Y, np.zeros((len(class_names), batch_size))), axis=1)
            y = np.concatenate((train_y,np.zeros((batch_size,))))

            # Solve
            X = solve(Y_compr, M, laplacian, alpha = alpha, beta = beta)

            # Make label vector
            x_hat = np.argmax(X,axis = 0) + np.ones(X[0,:].shape)

            # Unify labels 13-30
            x_hat_adapted = adapt_labels(x_hat)
            true_y = np.concatenate((train_y,label_vec[batch_index]))
            y_adapted = adapt_labels(true_y)

            # Only consider unknowns
            accuracy_tot.append(np.sum(x_hat[(len(x_hat)-batch_size):]==true_y[(len(x_hat)-batch_size):])/batch_size) # for all 30 words
            accuracy.append(np.sum(x_hat_adapted[(len(x_hat)-batch_size):]==y_adapted[(len(x_hat)-batch_size):])/batch_size) # only core words

            return accuracy, accuracy_tot
