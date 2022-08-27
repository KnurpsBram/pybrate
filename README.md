# fvrt audio
My favourite audio toolbox

### TODO:
- [ ] documentation

### Installation
```
pip install git+https://github.com/KnurpsBram/fvrt_audio
```
### Usage
```
import torch
from fvrt.fvrt_stft import FVRTSTFT

fvrt_stft = FVRTSTFT()
audio     = torch.randn(1, 16000)
spect     = fvrt_stft(audio)
```
