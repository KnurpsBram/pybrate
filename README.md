# fvrt audio
My favourite audio toolbox

### TODO:
- [ ] documentation

### Installation
```
git clone https://github.com/KnurpsBram/fvrt_audio
cd fvrt_audio
pip install .
```
You can remove the repo after installation.
### Usage
```
import torch
from fvrt.fvrt_stft import FVRTSTFT

fvrt_stft = FVRTSTFT()
audio     = torch.randn(1, 16000)
spect     = fvrt_stft(audio)
```
