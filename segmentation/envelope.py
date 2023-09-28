import numpy as np
from scipy import signal


def calc_analytic(data):
    analytic = signal.hilbert(data, axis=1)
    amp = np.absolute(analytic)
    freq = np.angle(analytic)
    return np.asarray(np.real(amp), float), np.asarray(np.real(freq), float)


def calc_energy(amp):
    energy = np.square(amp)
    ker = (1 / 7) * np.ones((1, 7))
    energy = signal.convolve(energy, ker, mode='same')
    return energy


def calc_envelope(wav_path):
    data = np.load(wav_path+'data.npy')
    sr = np.load(wav_path + 'sr.npy')
    data = np.squeeze(data[0, :, :])
    amp, freq = calc_analytic(data)
    energy = calc_energy(amp)
    return energy
