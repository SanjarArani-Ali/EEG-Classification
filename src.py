import numpy as np
import matplotlib.pyplot as plt
import pyedflib
from matplotlib.collections import LineCollection
from matplotlib.pyplot import figure
import mne
from tqdm.notebook import tqdm

from sklearn import preprocessing
from scipy import signal, stats
import pandas as pd



def feature_extract(inputsignal):
    features=[]
    analytic = signal.hilbert(inputsignal)
    features.extend([np.mean(np.abs(analytic)),
                     np.mean(np.diff(np.unwrap(np.angle(analytic)))) * 160 / (2 * np.pi), 
                     np.mean(np.angle(analytic)) ])
    
    features.extend([ np.ptp(inputsignal),
                     np.min(inputsignal),
                     np.max(inputsignal) ])
    
    rounded = np.round(inputsignal, 3)
    features.extend([np.median(inputsignal),
                     stats.mode(rounded)[0],
                     np.mean(inputsignal)])
    
    features.extend([np.sqrt(np.mean(inputsignal**2)),
                     np.std(inputsignal),
                     np.sum(inputsignal)])
    
    features.extend([np.var(inputsignal),
                     stats.kurtosis(inputsignal),
                     stats.skew(inputsignal)])

    return features

def create_columns_name(selectedLabels):
    columns_name=["Classes"]
    for electrod in range(len(selectedLabels)):
        for feature in ['Amplitude', 'Frequency', 'Phase', 'PeakToPeak', 
                        'NegativePeak', 'PositivePeak', 'Median', 'Mode', 
                        'ArithmeticMean', 'RMS', 'StandardDev' ,'Summation'
                        , 'Variance', 'Kurtoris', 'Skewness' ]:
           columns_name.append(f"{selectedLabels[electrod]}_{feature}")
    return columns_name

def load_EDF_File(subjectAddress,taskAddress):
    baseAddress= "./datasets/eegmmidb"
    fileAddress=f"{baseAddress}/S{subjectAddress}/S{subjectAddress}R{taskAddress}.edf"
    file = pyedflib.EdfReader(fileAddress)
    return file

def select_Data(file,selectedLabels,signal_labels):
    selectedData = np.zeros((len(selectedLabels), file.getNSamples()[0]))
    for i in np.arange(len(selectedLabels)):
        index=signal_labels.index(selectedLabels[i])
        file.readSignal(index)
        selectedData[i, :] = file.readSignal(index)
        
    return selectedData

def remove_Last_zero(annotation,sampleFrequency,selectedData):
    singalDuration= int((annotation[1] * sampleFrequency).sum())
    selectedData= selectedData [:,:singalDuration]
    
    return selectedData
    
def preproccessing_data(selectedData,selectedLabels,sampleFrequency):
    info = mne.create_info(ch_names=selectedLabels, sfreq=sampleFrequency,
                           ch_types='eeg')
    
    raw = mne.io.RawArray(selectedData.copy(), info,verbose='error')

    raw.filter(l_freq=0.1, h_freq=50.0,verbose='error')
    raw.notch_filter(freqs=50,verbose='error') 

    filtered_selectedData = np.array(raw.get_data())
    
    return filtered_selectedData

def find_Start_End_time(annotation,sampleFrequency,
                        filtered_selectedData,event_index):
    
    startTime = int(annotation[0][event_index]*sampleFrequency)
            
    if event_index+1<len(annotation[0]):
        endTime= int(annotation[0][event_index+1]*sampleFrequency)
    else:
        endTime= int(filtered_selectedData.shape[1])
    
    return startTime,endTime

def find_Class_Signal(annotation,task,event_index):
    if annotation[2][event_index] == "T0":
        class_signal=0
    elif task in [3,7,11]:
        if annotation[2][event_index]=="T1":
            class_signal=1
        elif annotation[2][event_index]=="T2":
            class_signal=2
    elif task in [5,9,13]:
        if annotation[2][event_index]=="T1":
            class_signal=3
        elif annotation[2][event_index]=="T2":
            class_signal=4
    return class_signal

def filter_data(thisData,selectedLabels,sampleFrequency,l_freq,h_freq):

    infoBand = mne.create_info(ch_names=selectedLabels, sfreq=sampleFrequency,
                               ch_types='eeg')
    
    rawBand = mne.io.RawArray(thisData.copy(), infoBand,verbose='error')
    rawBand.filter(l_freq=l_freq, h_freq=h_freq,verbose='error')
    inputsignal = np.array(rawBand.get_data())        
    
    return inputsignal

def balance_Between_Classes(trainData,taskfeature):
    taskfeature=np.array(taskfeature)
    lenRelax=taskfeature[taskfeature[:,0]==0].shape[0]
    relax_datas = taskfeature[taskfeature[:, 0] == 0]
    random_selected_relax_datas = relax_datas[np.random.choice(relax_datas.shape[0],
                                                               lenRelax//2+1,
                                                               replace=False)]
    
    not_relax_data = taskfeature[taskfeature[:, 0] != 0]
    taskfeature = np.vstack((random_selected_relax_datas, not_relax_data))
    trainData.extend(taskfeature)

    return trainData 

def extract_X_y_from_trainData(trainData,columns_name):
    trainData=np.array(trainData)
    trainDataPandas={}
    index=0
    for feature in columns_name:
        trainDataPandas[feature]=trainData[:,index]
        index+=1
        
    trainData = pd.DataFrame(trainDataPandas)
    
    X = trainData.drop(columns=['Classes'])
    y = trainData['Classes']
    
    return X,y

def select_Random_subject(nsubject):
    subjects= list(range (1,109+1)) 
    np.random.shuffle(subjects)
    
    return subjects[:nsubject]
    
def readFile_Preprocessing_ExtractFeature(tasks,selectedLabels,
                                          nsubject,randomseed):
    np.random.seed(randomseed)
    trainData=[]    
    columns_name= create_columns_name(selectedLabels)   
    Bands={
        0:(0.1,7.99),
        1:(4,11.99),
        2:(8,29.99),
        3:(12,49.99),
        4:(30,49.99)
    }       
    
    subjects= select_Random_subject(nsubject)
    
    for subject in tqdm(subjects,desc="subject"):
        for task in tqdm(tasks,desc="task", leave=False):
            taskfeature=[]
            
            subjectAddress= str(subject).zfill(3)
            taskAddress=  str(task).zfill(2)
            file= load_EDF_File(subjectAddress,taskAddress)
            
            nch = file.signals_in_file
            annotation = file.readAnnotations()
            PhysicalDimension=file.getPhysicalDimension(1)
            signal_labels = file.getSignalLabels()
            sampleFrequency= file.getSampleFrequency(0)
            
            signal_labels=[label.replace(".","") for label in signal_labels]
            selectedData = select_Data(file,selectedLabels,signal_labels)
            
            file.close()
            remove_Last_zero(annotation,sampleFrequency,selectedData)
            filtered_selectedData = preproccessing_data(selectedData,
                                                        selectedLabels,
                                                        sampleFrequency)
            
            for event_index in range(len(annotation[0])):
                
                startTime,endTime= find_Start_End_time(annotation,
                                                       sampleFrequency,
                                                       filtered_selectedData,
                                                       event_index)
                
                thisData=filtered_selectedData[:,startTime:endTime].copy()

                class_signal= find_Class_Signal(annotation,task,event_index)
                        
                l_freq,h_freq=Bands[class_signal]
                
                inputsignal= filter_data(thisData,selectedLabels,
                                         sampleFrequency,l_freq,h_freq)    
                
                eventfeature=[class_signal]
                for electrod in range(len(selectedLabels)):
                    eventfeature.extend(feature_extract(inputsignal[electrod]))    

                taskfeature.append(eventfeature)
            
            trainData = balance_Between_Classes(trainData,taskfeature)
    
    
    X,y= extract_X_y_from_trainData(trainData,columns_name)

    return X,y