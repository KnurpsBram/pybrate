import math
import torch

import pybrate


class TestGenerateAudio():

    frequency = 1
    sample_rate = 8
    seconds = 2
    angle_offset = math.pi/2
    sample_offset = 2

    sine_wave = torch.tensor([0., math.sqrt(0.5), 1., math.sqrt(0.5), 0., -math.sqrt(0.5), -1., -math.sqrt(0.5)]).repeat(2)
    square_wave = torch.tensor([1., 1., 1., 1., -1., -1., -1., -1.]).repeat(2)    
    sawtooth = torch.tensor([-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75]).repeat(2)
    pulse_train = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]).repeat(2)

    def test_generate_sine_wave(self):
        self._test_generate_waveform_fn(pybrate.generate_sine_wave, self.sine_wave)
        self._test_generate_waveform_fn_with_offset(pybrate.generate_sine_wave, self.sine_wave)

    def test_generate_square_wave(self):
        self._test_generate_waveform_fn(pybrate.generate_square_wave, self.square_wave)
        self._test_generate_waveform_fn_with_offset(pybrate.generate_square_wave, self.square_wave)

    def test_generate_sawtooth(self):
        self._test_generate_waveform_fn(pybrate.generate_sawtooth, self.sawtooth)
        self._test_generate_waveform_fn_with_offset(pybrate.generate_sawtooth, self.sawtooth)

    def test_generate_pulse_train(self):
        self._test_generate_waveform_fn(pybrate.generate_pulse_train, self.pulse_train)
        self._test_generate_waveform_fn_with_offset(pybrate.generate_pulse_train, self.pulse_train)

    def _test_generate_waveform_fn(self, fn, expected_waveform):
        assert torch.allclose(
            fn(self.frequency, self.sample_rate, seconds=self.seconds), 
            expected_waveform, 
            atol=1e-6
        )
    
    def _test_generate_waveform_fn_with_offset(self, fn, expected_waveform):
        assert torch.allclose(
            fn(self.frequency, self.sample_rate, seconds=self.seconds, angle_offset=self.angle_offset), 
            self._shift(expected_waveform, self.sample_offset), 
            atol=1e-6
        )

    def _shift(self, tensor, n):  # this function is only valid if the wave is exactly a whole number of periods
        """
        [1, 2, 3, 4] --> [4, 1, 2, 3]
        """
        return torch.cat([tensor[n:], tensor[:n]])
    