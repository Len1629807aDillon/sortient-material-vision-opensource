"""Dashboard rendering for analytics."""

from __future__ import annotations

from .monitoring import PerformanceMonitor


def render_console_dashboard(monitor: PerformanceMonitor) -> str:
    summary = monitor.summary()
    lines = ["Material Vision AI - Live Performance"]
    lines.append("=" * 48)
    for name, stat in summary.items():
        lines.append(f"{name:20s}: {stat.value:8.4f} (window={stat.window})")
    if len(lines) == 2:
        lines.append("No data yet. Start streaming metrics to populate the dashboard.")
    return "\n".join(lines)
