from typing import Dict, List, Tuple

def sid_to_pid_cid(sid: str, num_providers: int) -> str:
    """Normalize service IDs to provider/consumer form.

    If the ID already begins with ``"p"`` or ``"c"`` it is returned as-is.
    Otherwise, it is assumed to be of the form ``"S<number>"`` and is mapped
    to ``"p<number>"`` if the index refers to a provider or ``"c<number>``
    for consumers.
    """
    if sid.startswith("p") or sid.startswith("c"):
        return sid
    k = int(sid[1:])
    return f"p{k}" if k <= num_providers else f"c{k - num_providers}"

class StreakTracker:
    """
    Keeps tuples (producer_id, run_length) per consumer per time.
    """
    def __init__(self, consumer_ids: List[str], num_times: int):
        self.store: Dict[str, Dict[int, List[Tuple[str,int]]]] = {
            c: {t: [] for t in range(num_times)} for c in consumer_ids
        }
        self.marker: Dict[str, List[str]] = {c: [] for c in consumer_ids}

    def update(self, t: int, c_id: str, p_id: str):
        time_streaks = self.store[c_id]
        if t > 0:
            time_streaks[t] = time_streaks[t-1].copy()
        else:
            time_streaks[t] = []

        prev = self.marker[c_id][-1] if self.marker[c_id] else None
        if prev == p_id and time_streaks[t]:
            for i in range(len(time_streaks[t]) - 1, -1, -1):
                prod, count = time_streaks[t][i]
                if prod == p_id:
                    time_streaks[t][i] = (prod, count + 1)
                    break
        else:
            time_streaks[t].append((p_id, 1))
        self.marker[c_id].append(p_id)

    def get_at_time(self, t: int) -> Dict[str, List[Tuple[str,int]]]:
        return {c: time_dict.get(t, []) for c, time_dict in self.store.items()}
