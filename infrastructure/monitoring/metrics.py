"""Metrics collection utilities for the monitoring platform.

The metrics layer provides a thin abstraction over ``prometheus_client`` so the
rest of the codebase can interact with counters, gauges, histograms, and
summaries via a simple, thread-safe interface. The registry combines
application-level metrics with runtime statistics gathered elsewhere and offers
text-based exposition that can be scraped by Prometheus.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary
from prometheus_client.exposition import generate_latest


__all__ = [
    "MetricsRegistry",
    "MetricNotFoundError",
    "global_metrics",
]


class MetricNotFoundError(KeyError):
    """Raised when attempting to access a metric that is not registered."""


@dataclass
class _MetricDefinition:
    name: str
    documentation: str
    labels: Optional[Iterable[str]] = None


class MetricsRegistry:
    """Central registry for application metrics."""

    def __init__(self) -> None:
        self._registry = CollectorRegistry(auto_describe=False)
        self._metrics: Dict[str, object] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_counter(
        self, name: str, documentation: str, *, labelnames: Optional[Iterable[str]] = None
    ) -> Counter:
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, Counter):
                    return metric
                raise TypeError(f"Metric '{name}' already registered with different type")
            counter = Counter(
                name,
                documentation,
                labelnames=tuple(labelnames) if labelnames else (),
                registry=self._registry,
            )
            self._metrics[name] = counter
            return counter

    def register_gauge(
        self, name: str, documentation: str, *, labelnames: Optional[Iterable[str]] = None
    ) -> Gauge:
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, Gauge):
                    return metric
                raise TypeError(f"Metric '{name}' already registered with different type")
            gauge = Gauge(
                name,
                documentation,
                labelnames=tuple(labelnames) if labelnames else (),
                registry=self._registry,
            )
            self._metrics[name] = gauge
            return gauge

    def register_histogram(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: Optional[Iterable[str]] = None,
        buckets: Optional[Iterable[float]] = None,
    ) -> Histogram:
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, Histogram):
                    return metric
                raise TypeError(f"Metric '{name}' already registered with different type")
            histogram = Histogram(
                name,
                documentation,
                labelnames=tuple(labelnames) if labelnames else (),
                registry=self._registry,
                buckets=tuple(buckets) if buckets else Histogram.DEFAULT_BUCKETS,
            )
            self._metrics[name] = histogram
            return histogram

    def register_summary(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: Optional[Iterable[str]] = None,
        objectives: Optional[Dict[float, float]] = None,
    ) -> Summary:
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, Summary):
                    return metric
                raise TypeError(f"Metric '{name}' already registered with different type")
            summary = Summary(
                name,
                documentation,
                labelnames=tuple(labelnames) if labelnames else (),
                registry=self._registry,
                objectives=objectives,
            )
            self._metrics[name] = summary
            return summary

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def get(self, name: str) -> object:
        try:
            return self._metrics[name]
        except KeyError as exc:
            raise MetricNotFoundError(name) from exc

    def expose(self) -> bytes:
        """Return the Prometheus exposition format for all metrics."""

        with self._lock:
            return generate_latest(self._registry)

    def reset(self) -> None:
        """Clear registry and metrics (useful for tests)."""

        with self._lock:
            self._registry = CollectorRegistry(auto_describe=False)
            self._metrics.clear()


_global_registry = MetricsRegistry()


def global_metrics() -> MetricsRegistry:
    """Return the singleton metrics registry used by the application."""

    return _global_registry


@contextmanager
def track_duration(metric: Summary, **labels: str) -> Iterator[None]:
    """Context manager that records execution duration into a Summary metric."""

    if labels:
        timer = metric.labels(**labels).time()
    else:
        timer = metric.time()
    try:
        with timer:
            yield
    finally:
        pass

