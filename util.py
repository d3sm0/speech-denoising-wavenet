# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Util.py
# Utility functions for dealing with audio signals and training a Denoising Wavenet
import numpy as np
import json
import warnings
import scipy.signal
import scipy.stats
import soundfile as sf


def compute_receptive_field_length(stacks, dilations, filter_length, target_field_length=1):
    half_filter_length = (filter_length - 1) // 2
    length = 0
    for d in dilations:
        length += d * half_filter_length
    length = (stacks * 2 * length) + target_field_length
    return length


def snr_db(rms_amplitude_A, rms_amplitude_B):
    return 20.0 * np.log10(rms_amplitude_A / rms_amplitude_B)

#
# def wav_to_float(x):
#     # try:
#     max_value = np.iinfo(x.dtype).max
#     min_value = np.iinfo(x.dtype).min
#     # except:
#     #     max_value = np.finfo(x.dtype).max
#     #     min_value = np.finfo(x.dtype).min
#     x = x.astype('float64', casting='safe')
#     x -= min_value
#     x /= ((max_value - min_value) / 2.)
#     x -= 1.
#     return x


def one_hot_encode(x, num_values=256):
    # if isinstance(x, int):
    #     x = np.array([x])
    # if isinstance(x, list):
    #     x = np.array(x)
    return np.eye(num_values, dtype='uint8')  # [x.astype('uint8')]


def array_to_bin(x, max_value):
    assert isinstance(x, np.ndarray)
    width = np.ceil(np.log2(max_value)).astype(int)
    return ((x[:, None] & (1 << np.arange(width))) > 0).astype(int)


# def get_condition_input_encode_func(representation):
#     if representation == 'binary':
#         return array_to_bin
#     else:
#         return one_hot_encode


def pretty_json_dump(values, file_path=None):
    if file_path is None:
        print(json.dumps(values, sort_keys=True, indent=4, separators=(',', ': ')))
    else:
        json.dump(values, open(file_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


# def read_wav(filename):
#     # Reads in a wav audio file, takes the first channel, converts the signal to float64 representation
#     audio_signal, sample_rate = sf.read(filename, always_2d=True)
#     assert np.float64 == audio_signal.dtype
#
#     # if audio_signal.ndim > 1:
#     #     audio_signal = audio_signal[:, 0]
#
#     # if audio_signal.dtype != 'float64':
#     #     audio_signal = wav_to_float(audio_signal)
#
#     return audio_signal[:, 0], sample_rate


def load_wav(wav_path, desired_sample_rate=16000):

    seq, rate = sf.read(wav_path, always_2d=True)
    assert np.float64 == seq.dtype
    seq = seq[:, 0]

    if rate != desired_sample_rate:
        return scipy.signal.resample_poly(seq, desired_sample_rate, rate)

    return seq


def write_wav(x, filename, sample_rate):
    assert isinstance(x, np.ndarray)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf.write(filename, x, sample_rate)


def rms(x):
    return np.sqrt(np.mean(np.square(x), axis=-1))


def get_subsequence_with_speech_indices(full_sequence, chunk_length=800, num_pad_chunks=4):
    signal_magnitude = np.abs(full_sequence)

    chunks_energies = []
    for i in range(0, signal_magnitude.size, chunk_length):
        chunks_energies.append(np.mean(signal_magnitude[i:i + chunk_length]))

    chunks_energies = np.array(chunks_energies)
    th = np.max(chunks_energies) * .1

    onset_chunk_i = np.where(chunks_energies > th)[0][0]
    termination_chunk_i = np.where(chunks_energies > th)[0][-1]

    onset_chunk_i = np.max((0, onset_chunk_i - num_pad_chunks))
    termination_chunk_i = np.min((chunks_energies.size, termination_chunk_i + num_pad_chunks))

    return [onset_chunk_i * chunk_length, (termination_chunk_i + 1) * chunk_length]

#
# def dir_contains_files(path):
#     for f in os.listdir(path):
#         if not f.startswith('.'):
#             return True
#     return False

#
# def extract_subsequence_with_speech(full_sequence):
#     indices = get_subsequence_with_speech_indices(full_sequence)
#     return full_sequence[indices[0]:indices[1]]

#
# def normalize(x):
#     return x / np.max(np.abs(x))


# def float_to_uint8(x):
#     x += 1.
#     x /= 2.
#     uint8_max_value = np.iinfo('uint8').max
#     x *= uint8_max_value
#     x = x.astype('uint8')
#     return x


# def keras_float_to_uint8(x):
#     x += 1.
#     x /= 2.
#     uint8_max_value = 255
#     x *= uint8_max_value
#     return x
#
#
# def linear_to_ulaw(x, u=255):
#     x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
#     return x
#
#
# def keras_linear_to_ulaw(x, u=255.0):
#     x = keras.backend.sign(x) * (keras.backend.log(1 + u * keras.backend.abs(x)) / keras.backend.log(1 + u))
#     return x
#
# #
# def uint8_to_float(x):
#     max_value = np.iinfo('uint8').max
#     min_value = np.iinfo('uint8').min
#     x = x.astype('float32', casting='unsafe')
#     x -= min_value
#     x /= ((max_value - min_value) / 2.)
#     x -= 1.
#     return x

#
# def keras_uint8_to_float(x):
#     max_value = 255
#     min_value = 0
#     x -= min_value
#     x /= ((max_value - min_value) / 2.)
#     x -= 1.
#     return x

#
# def ulaw_to_linear(x, u=255.0):
#     y = np.sign(x) * (1 / float(u)) * (((1 + float(u)) ** np.abs(x)) - 1)
#     return y

#
# def keras_ulaw_to_linear(x, u=255.0):
#     y = keras.backend.sign(x) * (1 / u) * (((1 + u) ** keras.backend.abs(x)) - 1)
#     return y


#
# def one_hot_decode(x):
#     return np.argmax(x, axis=-1)

#
# def preemphasis(signal, alpha=0.95):
#     return np.append(signal[0], signal[1:] - alpha * signal[:-1])


#
# def ensure_keys_in_dict(keys, dictionary):
#     if all(key in dictionary for key in keys):
#         return True
#     return False

#
# def get_subdict_from_dict(keys, dictionary):
#     return dict((k, dictionary[k]) for k in keys if k in dictionary)
