# the code used to test FAD
https://github.com/gudgud96/frechet-audio-distance/tree/main
https://github.com/haoheliu/audioldm_eval

## please use a new environment to avoid conflict, you can use conda or uv
conda create -n svd python=3.10
cd audioldm_eval && pip install -e .
pip install nnAudio.
pip install resampy

## test FAD
cd audiometrics/
+ create folder `example`，then push the audios in `example`
+ set the `generation_result_path` and `target_audio_path` in `run.py`
+ run `run.py`
