import numpy as np
from dataclasses import dataclass

@dataclass
class RNGPool:
    master_seed: int
    num_times: int

    def __post_init__(self):
        root = np.random.SeedSequence(self.master_seed)
        # spawn core algorithm streams first so existing sequences remain stable
        ss_global, ss_init, ss_greedy, ss_nsga, ss_mopso, ss_mogwo, ss_scs = root.spawn(7)
        # spawn OU stream separately to preserve previous seeds
        ss_ou = root.spawn(1)[0]
        self.global_ = np.random.Generator(np.random.PCG64(ss_global))
        self.init = np.random.Generator(np.random.PCG64(ss_init))
        self.greedy = [np.random.Generator(np.random.PCG64(s)) for s in ss_greedy.spawn(self.num_times)]
        self.nsga   = [np.random.Generator(np.random.PCG64(s)) for s in ss_nsga.spawn(self.num_times)]
        self.mopso  = [np.random.Generator(np.random.PCG64(s)) for s in ss_mopso.spawn(self.num_times)]
        self.mogwo  = [np.random.Generator(np.random.PCG64(s)) for s in ss_mogwo.spawn(self.num_times)]
        self.scs    = [np.random.Generator(np.random.PCG64(s)) for s in ss_scs.spawn(self.num_times)]
        self.ou     = [np.random.Generator(np.random.PCG64(s)) for s in ss_ou.spawn(self.num_times)]
        self._ou_node = {}

    def for_(self, scope: str, t: int | None=None, idx: int | None=None) -> np.random.Generator:
        if scope == "greedy": return self.greedy[int(t)]
        if scope == "nsga":   return self.nsga[int(t)]
        if scope == "scs":
            if idx is not None:
                return np.random.Generator(
                    np.random.PCG64(
                        np.random.SeedSequence([self.master_seed, 3001, int(t), int(idx)])
                    )
                )
            return self.scs[int(t)]
        if scope == "ou":     return self.ou[int(t)]
        if scope in ("mopso","pso"): return self.mopso[int(t)]
        if scope in ("mogwo","gwo"): return self.mogwo[int(t)]
        if scope in ("init","ou_init"): return self.init
        if scope == "coords":
            return np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([self.master_seed, 2001, int(t)])
            ))
        if scope == "ou_node":
            idx = int(idx)
            if idx not in self._ou_node:
                self._ou_node[idx] = np.random.Generator(
                    np.random.PCG64(np.random.SeedSequence([self.master_seed, 1001, idx]))
                )
            return self._ou_node[idx]
        return self.global_

    def for_time(self, scope: str, t: int, idx: int | None = None) -> np.random.Generator:
        """Backward-compatible wrapper returning a per-time RNG."""
        return self.for_(scope, t, idx)
