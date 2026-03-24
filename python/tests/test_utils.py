"""Tests for CascadeGraph utility functions."""

import numpy as np
import pytest

from cascadegraph import (
    compute_filter,
    convolve_filter_with_stim,
    sample_nl,
    compute_variance_explained,
    apply_frequency_cutoff,
    apply_frequency_cutoff_to_fft,
    baseline_subtract,
)


class TestComputeFilter:
    def test_basic(self):
        np.random.seed(42)
        stim = np.random.randn(5, 1000)
        # Create a simple response: filtered version of stimulus
        true_filter = np.zeros(1000)
        true_filter[1] = 1.0  # delay by 1
        resp = np.real(np.fft.ifft(np.fft.fft(stim, axis=1) * np.fft.fft(true_filter), axis=1))

        fc, fa = compute_filter(stim, resp, 50)
        assert fc.shape == (50,)
        assert fa.shape == (50,)

    def test_with_frequency_cutoff(self):
        np.random.seed(42)
        stim = np.random.randn(3, 500)
        resp = np.random.randn(3, 500)
        fc, fa = compute_filter(stim, resp, 25, frequency_cutoff=20.0, sampling_interval=0.001)
        assert fc.shape == (25,)


class TestConvolveFilterWithStim:
    def test_shape(self):
        filt = np.zeros(100)
        filt[0] = 1.0
        stim = np.random.randn(3, 200)
        result = convolve_filter_with_stim(filt, stim, False)
        assert result.shape == (3, 200)

    def test_1d_input(self):
        filt = np.zeros(100)
        filt[0] = 1.0
        stim = np.random.randn(200)
        result = convolve_filter_with_stim(filt, stim, False)
        assert result.ndim == 1

    def test_odd_filter_raises(self):
        with pytest.raises(ValueError, match="even"):
            convolve_filter_with_stim(np.zeros(3), np.zeros(10), False)


class TestSampleNl:
    def test_equal_width(self):
        x = np.random.randn(100)
        y = 2 * x + 1
        nl_x, nl_y = sample_nl(x, y, 10, "equalWidth")
        assert len(nl_x) == 10
        assert len(nl_y) == 10

    def test_equal_n(self):
        x = np.random.randn(100)
        y = x ** 2
        nl_x, nl_y = sample_nl(x, y, 10, "equalN")
        assert len(nl_x) == 10

    def test_bad_bin_type(self):
        with pytest.raises(ValueError):
            sample_nl(np.zeros(10), np.zeros(10), 5, "invalid")


class TestComputeVarianceExplained:
    def test_perfect_prediction(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = compute_variance_explained(x, x)
        assert abs(r2 - 1.0) < 1e-10

    def test_2d(self):
        x = np.random.randn(3, 100)
        r2 = compute_variance_explained(x, x)
        np.testing.assert_allclose(r2, np.ones(3), atol=1e-10)

    def test_mismatched_shape(self):
        with pytest.raises(ValueError):
            compute_variance_explained(np.zeros((2, 3)), np.zeros((3, 3)))


class TestFrequencyCutoff:
    def test_apply_cutoff(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        filtered = apply_frequency_cutoff(signal, 10.0, 0.001)
        assert filtered.shape == signal.shape
        # Filtered signal should be smoother (lower variance)
        assert np.var(filtered) < np.var(signal)

    def test_2d(self):
        signal = np.random.randn(3, 200)
        filtered = apply_frequency_cutoff(signal, 10.0, 0.001)
        assert filtered.shape == signal.shape


class TestBaselineSubtract:
    def test_basic(self):
        data = np.array([[5.0, 5.0, 10.0, 15.0]])
        result = baseline_subtract(data, 2)
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0, 5.0, 10.0]])
