A Wavenet For Speech Denoising
====
This is a fork of the work done from [drethage](https://github.com/drethage/speech-denoising-wavenet).
The idea is to make it more efficient and ready to use for speech denosing of short duration.

Readme for complete usage will follow soon(ish).

The original paper and experiments can be found here:
A neural network for end-to-end speech denoising, as described in: "[A Wavenet For Speech Denoising](https://arxiv.org/abs/1706.07162)"

Listen to denoised samples under varying noise conditions and SNRs [here](http://www.jordipons.me/apps/speech-denoising-wavenet/)

Dataset
-----
The "Noisy speech database for training speech enhancement algorithms and TTS models" (NSDTSEA) is used for training the model. It is provided by the University of Edinburgh, School of Informatics, Centre for Speech Technology Research (CSTR).

1. [Download here](http://datashare.is.ed.ac.uk/handle/10283/1942)
2. Extract to `data/NSDTSEA`