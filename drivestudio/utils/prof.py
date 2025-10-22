# utils/prof.py
import os, re, time, cProfile, pstats, atexit
from contextlib import contextmanager
from pathlib import Path

def _parse_steps(expr: str):
    """
    expr examples: "100", "100-120", "100,200,300-320"
    returns a set() of ints (useful for small expr) or a callable for large ranges
    """
    if not expr:
        return None
    wanted = set()
    for part in expr.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        else:
            wanted.add(int(part))
    return wanted

class _Sink:
    """Collect multiple Stats and write a merged file at process exit."""
    def __init__(self, outdir: str, merged_name: str = "train_merged.pstats"):
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.merged = None
        self.merged_name = merged_name
        atexit.register(self._dump)

    def add(self, pstats_path: str):
        if self.merged is None:
            self.merged = pstats.Stats(pstats_path)
        else:
            self.merged.add(pstats_path)

    def _dump(self):
        if self.merged is not None:
            out = str(self.outdir / self.merged_name)
            self.merged.dump_stats(out)
            print(f"[cProfile] merged stats -> {out}")

_sink_cache = {}

def get_sink(outdir: str):
    if outdir not in _sink_cache:
        _sink_cache[outdir] = _Sink(outdir)
    return _sink_cache[outdir]

@contextmanager
def train_profiler(step: int,
                   enable: bool = False,
                   outdir: str = "profiles",
                   only_steps: str = "",
                   every_n: int | None = None,
                   cuda_sync_fn=None):
    """
    Use like: with train_profiler(step, enable=cfg.profiling.enable, only_steps="100-120,500", every_n=100, cuda_sync_fn=torch.cuda.synchronize):
        ... training block ...

    - enable: master on/off.
    - only_steps: profile specific steps/ranges.
    - every_n: profile every N steps (in addition to only_steps).
    - cuda_sync_fn: pass torch.cuda.synchronize to make GPU work visible to CPU profiler.
    """
    if not enable:
        yield
        return

    wanted = _parse_steps(only_steps)
    if wanted is not None and step not in wanted:
        if every_n is None or (step % every_n != 0):
            # skip
            yield
            return
    elif wanted is None and every_n is not None and (step % every_n != 0):
        # skip
        yield
        return

    Path(outdir).mkdir(parents=True, exist_ok=True)
    pr = cProfile.Profile()

    # Important for CUDA: flush pending kernels so CPU timing aligns with GPU work
    if cuda_sync_fn is not None:
        try: cuda_sync_fn()
        except Exception: pass

    pr.enable()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        pr.disable()
        if cuda_sync_fn is not None:
            try: cuda_sync_fn()
            except Exception: pass
        dt = (time.perf_counter() - t0) * 1000
        path = f"{outdir}/train_step_{int(step):06d}.pstats"
        pr.dump_stats(path)
        print(f"[cProfile] step {step} -> {path} ({dt:.1f} ms wall)")
        # add to merged sink
        get_sink(outdir).add(path)
