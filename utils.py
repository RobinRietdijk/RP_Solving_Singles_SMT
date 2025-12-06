def format_elapsed(elapsed: float) -> str:
    if elapsed < 1.0:
        ms = elapsed*1000
        return f"{ms:.3f} ms"
    else:
        return f"{elapsed:.3f} s"