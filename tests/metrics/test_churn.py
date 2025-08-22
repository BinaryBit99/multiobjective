from multiobjective.metrics import compute_churn


def test_compute_churn_simple():
    assignments = [[0, 1, 1], [1, 1, 0], [1, 0, 0]]
    churn = compute_churn(assignments)
    assert churn == [2/3, 1/3]


def test_compute_churn_single_step():
    assert compute_churn([[0, 1]]) == []
