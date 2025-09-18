from material_vision_ai.analytics.monitoring import PerformanceMonitor
from material_vision_ai.analytics.dashboard import render_console_dashboard


def test_monitoring_summary() -> None:
    monitor = PerformanceMonitor(window=5)
    for idx in range(5):
        monitor.update({"throughput": idx + 1, "accuracy": 0.9 + idx * 0.01})
    summary = monitor.summary()
    assert "throughput" in summary
    dashboard = render_console_dashboard(monitor)
    assert "Material Vision AI" in dashboard
