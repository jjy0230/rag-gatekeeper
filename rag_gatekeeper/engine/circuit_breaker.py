"""Production-grade circuit breaker for protecting downstream inference engines."""

import time
import logging
from typing import Callable, Any, Awaitable

logger = logging.getLogger(__name__)


class CircuitBreakerOpenException(Exception):
    """Raised when request is fast-failed because circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    """Stateful circuit breaker pattern implementation.
    
    States:
      - CLOSED: Requests pass through normally. Failures increment failure counter.
      - OPEN: Failures exceeded threshold. All requests fail immediately without network I/O.
      - HALF_OPEN: Recovery timeout elapsed. Single probe request allowed through to test health.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        name: str = "jev_engine",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.name = name

        self.state: str = "CLOSED"
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0

    def record_success(self):
        """Record successful invocation and reset breaker to CLOSED."""
        if self.state != "CLOSED":
            logger.info("Circuit breaker [%s] recovered to CLOSED state.", self.name)
        self.state = "CLOSED"
        self.failure_count = 0

    def record_failure(self):
        """Record failed invocation and trip breaker if threshold exceeded."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                "Circuit breaker [%s] TRIPPED to OPEN state (%d consecutive failures).",
                self.name,
                self.failure_count,
            )

    def before_call(self):
        """Check breaker state before allowing request through."""
        now = time.time()
        if self.state == "OPEN":
            if now - self.last_failure_time >= self.recovery_timeout_sec:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker [%s] entered HALF_OPEN state (probing...).", self.name)
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker [{self.name}] is OPEN. Fast-failing request."
                )

    async def call(self, coro_func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute coroutine with circuit breaker protection."""
        self.before_call()
        try:
            result = await coro_func()
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure()
            raise exc
