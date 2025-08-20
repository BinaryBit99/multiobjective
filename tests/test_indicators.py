import math
import pytest

from multiobjective.indicators import hypervolume_2d, igd, epsilon_additive


def test_indicators_on_simple_front():
    front = [(0.2, 0.3), (0.4, 0.1)]
    reference = [(0.0, 0.0), (0.2, 0.4)]

    hv = hypervolume_2d(front, ref=(1.0, 1.0))
    igd_val = igd(front, reference)
    eps_val = epsilon_additive(front, reference)

    expected_hv = (0.4 - 0.2) * (1.0 - 0.1)
    expected_igd = (math.sqrt(0.2 ** 2 + 0.3 ** 2) + 0.1) / 2
    expected_eps = 0.3

    assert hv == pytest.approx(expected_hv)
    assert igd_val == pytest.approx(expected_igd)
    assert eps_val == pytest.approx(expected_eps)
