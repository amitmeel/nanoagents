"""
Cancellation token implementation for graceful task cancellation.

This module provides a thread-safe cancellation mechanism following AutoGen's proven design
for cancelling pending async operations in a consistent manner.
"""

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Callable, List, Union


class CancellationToken:
    """A token used to cancel pending async calls"""

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        """Cancel pending async calls linked to this cancellation token."""
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                for callback in self._callbacks:
                    try:
                        callback()
                    except Exception:
                        # Don't let callback errors affect cancellation
                        pass

    def is_cancelled(self) -> bool:
        """Check if the CancellationToken has been used"""
        with self._lock:
            return self._cancelled

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Attach a callback that will be called when cancel is invoked"""
        with self._lock:
            if self._cancelled:
                try:
                    callback()
                except Exception:
                    # Don't let callback errors affect cancellation
                    pass
            else:
                self._callbacks.append(callback)

    def link_future(
        self, future: Union[Future[Any], asyncio.Future[Any], asyncio.Task[Any]]
    ) -> Union[Future[Any], asyncio.Future[Any], asyncio.Task[Any]]:
        """Link a pending async call to a token to allow its cancellation"""
        with self._lock:
            if self._cancelled:
                future.cancel()
            else:

                def _cancel() -> None:
                    future.cancel()

                self._callbacks.append(_cancel)
        return future
