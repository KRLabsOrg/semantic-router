"""Canonical text form of a signal vector.

Rules are learned and executed over this string rather than over the request, so
that a learned condition is a condition on signals — the same vocabulary the
decision engine matches on.
"""


def signal_text(row) -> str:
    """Render one row of `compute_signals.py` output as a stable string."""
    parts = [f"domain={row.get('domain') or 'none'}"]
    for key in sorted(row.keys()):
        if key.startswith("complexity::"):
            parts.append(f"{key.split('::', 1)[1]}={row[key]}")
    active = [
        key.split("::", 1)[1]
        for key in sorted(row.keys())
        if key.startswith(("structure::", "keyword::", "context::")) and bool(row[key])
    ]
    parts.append("signals=" + (",".join(active) if active else "none"))
    return " ".join(parts)
