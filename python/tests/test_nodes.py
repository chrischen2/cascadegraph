"""Tests for CascadeGraph node classes."""

import numpy as np
import pytest

from cascadegraph import (
    DataNode,
    FilterNode,
    ParamFilterNode,
    PolyfitNlNode,
    SigmoidNlNode,
    SumNode,
    NegativeNode,
    LnHyperNode,
    TwoArmLnHyperNode,
    RodLinearNode,
    RodBiophysNode,
)


class TestDataNode:
    def test_returns_stored_data(self):
        data = np.array([1.0, 2.0, 3.0])
        node = DataNode(data)
        result = node.process_upstream()
        np.testing.assert_array_equal(result, data)

    def test_raises_if_has_upstream(self):
        node = DataNode(np.array([1.0]))
        node.upstream.append(DataNode(np.array([2.0])))
        with pytest.raises(ValueError):
            node.process_upstream()


class TestFilterNode:
    def test_convolution(self):
        # Simple impulse filter
        filt = np.zeros(100)
        filt[0] = 1.0
        stim = np.random.randn(2, 100)
        node = FilterNode(filt, False)
        # Mean-subtracted stimulus convolved with impulse should be ~mean-subtracted stim
        result = node.process(stim)
        assert result.shape == stim.shape

    def test_graph_execution(self):
        filt = np.zeros(100)
        filt[0] = 1.0
        stim = np.random.randn(2, 100)
        stim_node = DataNode(stim)
        filt_node = FilterNode(filt, False)
        filt_node.upstream.append(stim_node)
        result = filt_node.process_upstream()
        assert result.shape == stim.shape


class TestParamFilterNode:
    def test_get_filter(self):
        params = np.array([3.0, 0.01, 0.03, 0.1, 0.0])
        node = ParamFilterNode(params)
        filt = node.get_filter(1000, 0.001)
        assert filt.shape == (1000,)
        # Filter should be normalized to max abs = 1
        assert abs(max(abs(filt.max()), abs(filt.min())) - 1.0) < 1e-10

    def test_process(self):
        params = np.array([3.0, 0.01, 0.03, 0.1, 0.0])
        node = ParamFilterNode(params)
        stim = np.random.randn(2, 500)
        result = node.process(stim, dt=0.001)
        assert result.shape == (2, 500)


class TestPolyfitNlNode:
    def test_fit_and_evaluate(self):
        x = np.linspace(-3, 3, 100)
        y = 2 * x**2 + 0.5 * x - 1 + np.random.randn(100) * 0.01
        node = PolyfitNlNode()
        node.fit_to_sample(x, y, degree=2)
        pred = node.process(x)
        # Should be a good fit
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0.99


class TestSigmoidNlNode:
    def test_sigmoid_shape(self):
        node = SigmoidNlNode(np.array([2.0, 1.0, 0.0, -1.0]))
        x = np.linspace(-5, 5, 100)
        y = node.process(x)
        # Should be monotonically increasing (normcdf is)
        assert np.all(np.diff(y) >= 0)

    def test_graph_execution(self):
        stim_node = DataNode(np.linspace(-3, 3, 100))
        nl_node = SigmoidNlNode(np.array([2.0, 1.0, 0.0, -1.0]))
        nl_node.upstream.append(stim_node)
        result = nl_node.process_upstream()
        assert result.shape == (100,)


class TestSumNode:
    def test_sums_two_inputs(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        node = SumNode()
        result = node.return_output([a, b])
        np.testing.assert_array_equal(result, a + b)

    def test_graph_sum(self):
        d1 = DataNode(np.array([1.0, 2.0]))
        d2 = DataNode(np.array([3.0, 4.0]))
        s = SumNode()
        s.upstream.append(d1)
        s.upstream.append(d2)
        result = s.process_upstream()
        np.testing.assert_array_equal(result, np.array([4.0, 6.0]))


class TestNegativeNode:
    def test_negates(self):
        d = DataNode(np.array([1.0, -2.0, 3.0]))
        n = NegativeNode()
        n.upstream.append(d)
        result = n.process_upstream()
        np.testing.assert_array_equal(result, np.array([-1.0, 2.0, -3.0]))


class TestLnHyperNode:
    def test_process(self):
        params = np.array([3.0, 0.01, 0.03, 0.1, 0.0, 2.0, 0.1, -1.0, -1.0])
        model = LnHyperNode(params)
        model.dt_stored = 0.001
        stim = np.random.randn(1, 500)
        result = model.process(stim, dt=0.001)
        assert result.shape == (1, 500)

    def test_graph_execution(self):
        params = np.array([3.0, 0.01, 0.03, 0.1, 0.0, 2.0, 0.1, -1.0, -1.0])
        model = LnHyperNode(params)
        model.dt_stored = 0.001
        stim_node = DataNode(np.random.randn(1, 500))
        model.upstream.append(stim_node)
        result = model.process_upstream()
        assert result.shape == (1, 500)


class TestRodLinearNode:
    def test_process(self):
        node = RodLinearNode(
            np.array([1.0, 0.01, 0.03]),
            other_params={"darkCurrent": 10.0},
        )
        stim = np.random.randn(500)
        result = node.process(stim, dt=0.001)
        assert result.shape == (500,)


class TestRodBiophysNode:
    def test_process(self):
        node = RodBiophysNode(
            np.array([25.0, 0.4, 100.0, 10.0, 2000.0]),
            other_params={"betaSlow": 0.4, "hillcoef": 4.0, "darkCurrent": 20.0},
        )
        stim = np.zeros(1000)
        stim[100:110] = 100.0  # flash
        result = node.process(stim, dt=0.001)
        assert result.shape == (1000,)


class TestGraphComposition:
    """Test multi-node graph construction and execution."""

    def test_simple_ln_cascade(self):
        """Build DataNode -> FilterNode -> PolyfitNlNode manually."""
        stim = np.random.randn(2, 200)
        filt = np.zeros(200)
        filt[0] = 1.0

        stim_node = DataNode(stim)
        filt_node = FilterNode(filt, False)
        filt_node.upstream.append(stim_node)

        # Fit a polynomial to identity
        nl_node = PolyfitNlNode()
        nl_node.fit_to_sample(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50), degree=1)
        nl_node.upstream.append(filt_node)

        result = nl_node.process_upstream()
        assert result.shape == (2, 200)
