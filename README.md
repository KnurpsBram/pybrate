# Pybrate
An audio toolbox built for PyTorch

### Installation
```
pip install git+https://github.com/KnurpsBram/pybrate
```
### Usage
One nice functionality of pybrate is that the default behavior of `stft` returns an stft that has exactly one column per `hop_length` waveform samples

```
import pybrate

win_length = 1024
hop_length = 256

my_audio = torch.randn(1, 100*hop_length)
my_stft = pybrate.stft(my_audio, win_length, hop_length)

print(my_stft.shape)
>>> [1, 513, 100]
```

### TODO:
- [ ] more unit tests
- [ ] documentation
