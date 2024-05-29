import pandas as pd
import numpy as np
import wave
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import write
from scipy import signal

def read_wav_file(filename):
    with wave.open(filename, 'rb') as wf:
        params = wf.getparams()
        num_channels, sampwidth, framerate, num_frames = params[:4]
        frames = wf.readframes(num_frames)
        waveform = np.frombuffer(frames, dtype=np.int16)
    return waveform, params

def read_wav_file_scipy(filename):
    framerate, waveform = wavfile.read(filename)
    return waveform, framerate

def plot_waveform(waveform, framerate):
    # Create a time array in seconds
    time_array = np.arange(0, len(waveform)) / framerate
    plt.figure(figsize=(15, 5))
    plt.plot(time_array, waveform, label="Waveform")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.legend()
    plt.show()

def read_files_in_dir(directory):
    filenames = os.listdir(directory)
    return filenames


def pick_5_samples(arrays):
    instruments = []
    for array in arrays:
        pick = np.random.choice(array, 1)
        instruments.append(pick)
    return instruments

def pick_samples_and_classify(arrays):
    #Picks a random number of samples, and returns their filepath and label
    instruments = []
    #pick at minimum two instruments
    number_of_instruments = np.random.randint(2, len(arrays) + 1)
    labels = np.zeros(len(arrays))
    already_picked = []

    while len(instruments) < number_of_instruments:
        random_pick = np.random.randint(0, len(arrays))
        if random_pick in already_picked:
            break
        else:
            already_picked.append(random_pick)
            pick = np.random.choice(arrays[random_pick], 1)
            instruments.append(pick)
            labels[random_pick] = 1

    return instruments, labels

#read the filenames, and add their data to 5 lists
def add_waveform_to_list(filenames, path = "./audio/"):
    waveforms = []
    for filename in filenames:
        waveform, params = read_wav_file_scipy(path + filename[0])
        waveforms.append(waveform)
    return waveforms
        
#Fast fourier transform
def fft_h(data, sample_rate):
    n = len(data)
    fft_data = np.fft.fft(data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    return freq[:n//2], np.abs(fft_data[:n//2])

def combine_waveforms(waveforms):
    normalization = 1 / len(waveforms)
    out = np.zeros_like(waveforms[0], dtype=np.float32)
    for w in waveforms:
        out += w.astype(np.float32) * normalization
    return out # note, this retuns a float32 array - it is needed to convert this to int16 before saving it to a wav file


def waveform_to_wavfile(waveform, name_string, sample_rate = 16000):
    write(name_string, sample_rate, waveform.astype(np.int16))

def create_spectrogram(waveform, sample_rate = 16000):
    freq, ts, spectro = signal.spectrogram(waveform, sample_rate)
    return freq, ts, spectro

def plot_spectrogram(freq, ts, spectrogram):
    plt.figure(figsize=(15, 5))
    plt.pcolormesh(ts, freq, spectrogram, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.show()

def gen_combo_waveform(filenames):
    pianos = [filename for filename in filenames if "piano" in filename] #empty
    bass = [filename for filename in filenames if "bass" in filename]
    guitar = [filename for filename in filenames if "guitar" in filename]
    drum = [filename for filename in filenames if "drum" in filename] #empty
    flutes = [filename for filename in filenames if "flute" in filename]
    keyboards = [filename for filename in filenames if "keyboard" in filename]
    paths, label = pick_samples_and_classify([pianos, bass, guitar, drum, flutes, keyboards])
    waveforms = add_waveform_to_list(paths)
    return combine_waveforms(waveforms), label

def gen_data_set(N):
    data = []
    labels = []
    for i in range(N):
        waveform, label = gen_combo_waveform()
        
        data.append(waveform)
        labels.append(label)
    return data, labels

def gen_spectrogram_set(N):
    data = []
    labels = []
    for i in range(N):
        waveform, label = gen_combo_waveform()
        freq, ts, spectro = create_spectrogram(waveform)
        data.append(spectro)
        labels.append(label)
    return data, labels
#Sorting the files in directory

def set_file_path(filepath):
    path = filepath
    return path

def create_single_inst_classification_set(N_samples = 1000, classes = ['bass', 'flute', 'guitar', 'keyboard'], path = "./audio/"):
    filenames = read_files_in_dir(path)
    data_spectogrmas = []
    labels = []
    for i in range(N_samples):
        # Pick random instrument
        random_int = np.random.randint(0, len(classes))
        instrument = classes[random_int]
        # Find all files with that instrument
        instrument_files = [filename for filename in filenames if instrument[0] in filename]
        # Pick a random file
        audio_file = np.random.choice(instrument_files, 1)
        # Read the file
        waveform, framerate = read_wav_file_scipy(path + audio_file[0])
        # Create spectrogram
        freq, ts, spectrogram = create_spectrogram(waveform, framerate)
        # Flatten the spectrogram and append to data
        data_spectogrmas.append(spectrogram.flatten())
        labels.append(random_int)

    # Create a label dictionary
    label_dict = {}
    for i, label in enumerate(classes):
        label_dict[i] = label

    # Convert to numpy arrays
    data_spectogrmas = np.array(data_spectogrmas)
    labels = np.array(labels)

    return data_spectogrmas, labels, label_dict