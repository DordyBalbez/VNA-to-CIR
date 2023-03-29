import matplotlib.pyplot as plt
import math as m
import numpy as np
import pandas as pd
import scipy as sp
import decimal
from scipy.signal import hilbert

plt.style.use('dark_background')

def get_frequency_time(file):
    matrix = np.genfromtxt(file, dtype=float, delimiter=',', skip_header=3)
    frequency = matrix[:, 0]
    frequency = np.reshape(frequency, (len(frequency), 1))
    bandwidth = frequency[-1] - frequency[0]
    time = np.linspace(0, len(frequency) / (2 * bandwidth), 20*len(frequency)).reshape(1, 20*len(frequency))
    return frequency, time


def get_data(file):
    read = pd.read_csv(file, delimiter=',', skiprows=2)
    column_index = read.columns.get_indexer(['re:Trc3_S42', 'im:Trc3_S42', 're:Trc3_S24', 'im:Trc3_S24'])
    column_index = column_index[column_index > 0]
    if len(column_index) == 0:
        print("No S24 or S42 data in the data set.")
        exit
    column_real, column_im = column_index
    data = np.genfromtxt(file, dtype=complex, delimiter=',', skip_header=3)
    data = data[:, column_real] + 1j * data[:, column_im]
    data = np.reshape(data, (len(data), 1))
    return data


def ift_manual(data, frequency, time):
    df = abs(frequency[2] - frequency[1])
    if len(frequency) != 1:
        frequency = frequency.T
    if len(data) != len(frequency):
        data = data.T
    data = data @ np.exp(2j * np.pi * frequency.T @ time) * df
    return data

def debug(file):
    frequency, time = get_frequency_time(file)
    data = get_data(file)
    power = get_power(data, frequency, time)
    xlim, time_short, power_short = get_xlim(time, power)
    return frequency, time, power, time_short, power_short, xlim

def get_power(data, frequency, time):
    y_ift = 2 * ift_manual(data, frequency, time)
    h = np.squeeze(y_ift.real)
    power = abs(hilbert(h)) ** 2
    return power


def get_xlim(time, power):
    time = time.reshape(power.shape)
    time_short = time[power > max(power) / 100]
    xlim = time_short[-1]
    power_short = power[0:len(time_short):1]
    return xlim, time_short, power_short


def cir(file):
    frequency, time = get_frequency_time(file)
    data = get_data(file)
    power = get_power(data, frequency, time)
    xlim, time_short, power_short = get_xlim(time, power)
    plt.plot(np.squeeze(time), power)
    plt.xlim(0, xlim)
    plt.xlabel('Time (s)')
    plt.ylabel('Power of CIR')
    plt.text(0.75*xlim, 0.75*max(power), papr(power))
    plt.text(0.75 * xlim, 0.6 * max(power), pp2p(power))
    plt.title(file)


def pp2p(power):
    peak_index, _ = sp.signal.find_peaks(power)
    peaks = []
    for i in range(len(peak_index)):
        peaks.append(power[peak_index[i]])
    # max_peak_index = np.where(power[peak_index] == max(power))
    # max_peak_index = int(max_peak_index)
    max_peak_index = np.argmax(power)
    peak_before_max = int(peak_index[max_peak_index - 1])
    peak_after_max = int(peak_index[max_peak_index + 1])
    try:
        a, b = sp.signal.argrelmin(power[peak_before_max: peak_after_max: 1])[0]
        numerator = np.trapz(power[peak_before_max + a: peak_after_max - b: 1])
        denominator = np.trapz(power)                                                     # maybe change to power?
    except ValueError:
        a = max_peak_index
        b = int(sp.signal.argrelmin(power[a: peak_after_max: 1])[0])
        numerator = 2 * np.trapz(power[a: peak_after_max - b: 1])
        denominator = np.trapz(power)
    raysh = '%.2e' % decimal.Decimal(numerator / denominator)
    string = "$Ratio=$" + str(raysh)
    return string

def papr(power):
    papr = 10*m.log10(max(power) / sp.mean(power))
    paper = round(papr, 3)
    string = "$PAPR=$" + str(paper)
    return string
