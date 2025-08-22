from collections.abc import Sequence


def compute_churn(assignments: Sequence[Sequence[int]]) -> list[float]:
    """Compute consumer-provider churn between consecutive time steps.

    Parameters
    ----------
    assignments : sequence of sequences of int
        ``assignments[t][i]`` denotes the provider index serving consumer ``i``
        at time ``t``.

    Returns
    -------
    list of float
        Fraction of consumers whose provider changes between ``t`` and
        ``t+1`` for each consecutive pair of assignments.
    """
    churn: list[float] = []
    for prev, curr in zip(assignments[:-1], assignments[1:]):
        if not prev:
            churn.append(0.0)
            continue
        changes = sum(1 for a, b in zip(prev, curr) if a != b)
        churn.append(changes / len(prev))
    return churn
