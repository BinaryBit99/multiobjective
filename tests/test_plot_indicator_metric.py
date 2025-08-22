import matplotlib
matplotlib.use("Agg")

from multiobjective.plotting import indicator_metric_series, plot_indicator_metric


def test_indicator_series_and_plot():
    indicators = {
        "alg1": {"tp": {"HV": [0.1, 0.2]}},
        "alg2": {"tp": {"HV": [0.3, 0.4]}},
    }

    times, series, labels = indicator_metric_series(indicators, "HV", "tp")
    assert times == [0, 1]
    assert series == [[0.1, 0.2], [0.3, 0.4]]
    assert labels == ["alg1", "alg2"]

    # Should plot without raising an error
    plot_indicator_metric(indicators, "HV", "tp", "HV", "HV")
