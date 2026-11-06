"""Audio landmarks for pin snapping: speaking peaks and pauses from the volume envelope of the speech.

A pin marked ~s (the picture shows a SPEAKING mouth shape) snaps to the nearest energy peak; a pin marked ~c (closed /
neutral mouth) snaps to the nearest pause. Times are seconds into the speech; the caller adds the lead-in.
"""
import numpy as np

HOP = 0.01          # envelope step, seconds
PAUSE_LEVEL = 0.08  # fraction of the loudest envelope value
PAUSE_MIN = 0.15    # shortest pause that counts, seconds
PEAK_LEVEL = 0.45
PEAK_SPACING = 0.15


def envelope(waveform, sr):
    """-> (env 0..1 smoothed RMS per HOP, hop seconds). waveform: array [C, T] or [T]."""
    x = np.asarray(waveform, dtype=np.float32)
    x = x.mean(axis=0) if x.ndim == 2 else x
    hop = max(int(HOP * sr), 1)
    n = len(x) // hop
    if n < 8:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    env = np.convolve(rms, np.ones(7) / 7, mode="same")
    return env / (env.max() + 1e-9)


def landmarks(env):
    """-> (peaks, pauses) as lists of seconds."""
    n = len(env)
    pauses, i = [], 0
    low = env < PAUSE_LEVEL
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if (j - i) * HOP >= PAUSE_MIN:
                pauses.append((i + j) / 2 * HOP)
            i = j
        else:
            i += 1
    w = int(PEAK_SPACING / HOP)
    peaks = [i * HOP for i in range(w, n - w) if env[i] > PEAK_LEVEL and env[i] == env[i - w: i + w + 1].max()]
    return peaks, pauses


def snap_time(t, mode, peaks, pauses, window=1.0):
    """mode 1 = nearest peak, 2 = nearest pause, within +-window seconds; otherwise t is returned unchanged."""
    pool = peaks if mode == 1 else pauses if mode == 2 else []
    near = [p for p in pool if abs(p - t) <= window]
    return min(near, key=lambda p: abs(p - t)) if near else t


def snap_indices(idx, modes, env, pad, total, fps=24):
    """idx / modes: lists of 8 (absolute frame in the generated clip, -1 = last). Returns the snapped list, spacing kept >= 8."""
    peaks, pauses = landmarks(env) if len(env) else ([], [])
    last = total - 1
    out = list(idx)
    for k, (v, m) in enumerate(zip(idx, modes)):
        if v < 0 or m == 0:
            continue
        t = (v - pad) / fps
        s = snap_time(t, m, peaks, pauses)
        out[k] = min(max(int(round(s * fps)) + pad, pad + 1), last)
    order = sorted((k for k, v in enumerate(out) if v >= 0), key=lambda k: (out[k], k))
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < 8:
            out[b] = min(out[a] + 8, last)
    return out
