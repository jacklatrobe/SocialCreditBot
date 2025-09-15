"""
Discord Error Handling and Resilience Module

This module provides comprehensive error handling, retry mechanisms, and circuit breaker
patterns for Discord operations. Implements robust recovery strategies for network
failures, rate limiting, authentication errors, and permission issues.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import discord
from discord.errors import (
    HTTPException, 
    RateLimited, 
    Forbidden, 
    NotFound, 
    DiscordServerError,
    ConnectionClosed,
    LoginFailure
)


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of Discord error types."""
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NETWORK = "network"
    SERVER_ERROR = "server_error"
    NOT_FOUND = "not_found"
    CONNECTION_LOST = "connection_lost"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, reject requests
    HALF_OPEN = "half_open"  # Testing if service is recovered


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.RATE_LIMIT,
        ErrorType.NETWORK,
        ErrorType.SERVER_ERROR,
        ErrorType.CONNECTION_LOST
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2  # For half-open state
    monitoring_window: float = 300.0  # 5 minutes


@dataclass
class ErrorMetrics:
    """Error tracking metrics."""
    total_errors: int = 0
    errors_by_type: Dict[ErrorType, int] = field(default_factory=dict)
    retry_attempts: int = 0
    circuit_breaker_trips: int = 0
    rate_limit_hits: int = 0
    last_error_time: Optional[datetime] = None
    error_rate_window: List[datetime] = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit breaker implementation for Discord operations.
    
    Prevents cascading failures by temporarily stopping requests to a failing service
    and automatically recovering when the service is healthy again.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_changed_time = datetime.now()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.state_changed_time = datetime.now()
        logger.info("🔄 Circuit breaker transitioned to HALF_OPEN")
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to open state."""
        self.state = CircuitState.OPEN
        self.state_changed_time = datetime.now()
        logger.warning(f"⚠️ Circuit breaker OPENED after {self.failure_count} failures")
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.state_changed_time = datetime.now()
        logger.info("✅ Circuit breaker CLOSED - service recovered")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'state_changed_time': self.state_changed_time.isoformat(),
            'time_in_current_state': (datetime.now() - self.state_changed_time).total_seconds()
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class DiscordErrorHandler:
    """
    Comprehensive error handling for Discord operations.
    
    Provides retry mechanisms, circuit breaker protection, rate limit handling,
    and comprehensive error classification and logging.
    """
    
    def __init__(
        self, 
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None
    ):
        """Initialize the error handler."""
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_config or CircuitBreakerConfig())
        self.metrics = ErrorMetrics()
    
    async def handle_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic and circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Various Discord exceptions after retry exhaustion
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Execute through circuit breaker
                result = await self.circuit_breaker.call(func, *args, **kwargs)
                
                # Reset retry count on success
                if attempt > 0:
                    logger.info(f"✅ Operation succeeded after {attempt + 1} attempts")
                
                return result
                
            except CircuitBreakerOpenError:
                self.metrics.circuit_breaker_trips += 1
                raise
                
            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)
                await self._handle_error(e, error_type, attempt + 1)
                
                # Check if error is retryable
                if error_type not in self.retry_config.retryable_errors:
                    logger.error(f"❌ Non-retryable error {error_type.value}: {e}")
                    raise
                
                # Check if this was the last attempt
                if attempt == self.retry_config.max_attempts - 1:
                    logger.error(f"❌ Max retry attempts ({self.retry_config.max_attempts}) exceeded")
                    raise
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, error_type)
                logger.warning(f"🔄 Retrying in {delay:.2f}s (attempt {attempt + 2}/{self.retry_config.max_attempts})")
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify a Discord error into a specific error type."""
        if isinstance(error, RateLimited):
            return ErrorType.RATE_LIMIT
        elif isinstance(error, LoginFailure):
            return ErrorType.AUTHENTICATION
        elif isinstance(error, Forbidden):
            return ErrorType.PERMISSION
        elif isinstance(error, NotFound):
            return ErrorType.NOT_FOUND
        elif isinstance(error, DiscordServerError):
            return ErrorType.SERVER_ERROR
        elif isinstance(error, ConnectionClosed):
            return ErrorType.CONNECTION_LOST
        elif isinstance(error, (OSError, asyncio.TimeoutError)):
            return ErrorType.NETWORK
        else:
            return ErrorType.UNKNOWN
    
    async def _handle_error(self, error: Exception, error_type: ErrorType, attempt: int):
        """Handle a specific error and update metrics."""
        self.metrics.total_errors += 1
        self.metrics.retry_attempts += 1
        self.metrics.last_error_time = datetime.now()
        
        # Update error type counters
        if error_type not in self.metrics.errors_by_type:
            self.metrics.errors_by_type[error_type] = 0
        self.metrics.errors_by_type[error_type] += 1
        
        # Update error rate window (keep last 100 errors for rate calculation)
        self.metrics.error_rate_window.append(self.metrics.last_error_time)
        if len(self.metrics.error_rate_window) > 100:
            self.metrics.error_rate_window.pop(0)
        
        # Handle specific error types
        if error_type == ErrorType.RATE_LIMIT:
            await self._handle_rate_limit(error)
        elif error_type == ErrorType.AUTHENTICATION:
            logger.critical(f"🚨 Authentication failed: {error}")
        elif error_type == ErrorType.PERMISSION:
            logger.error(f"🔒 Permission denied: {error}")
        elif error_type == ErrorType.CONNECTION_LOST:
            logger.warning(f"🔌 Connection lost: {error}")
        elif error_type == ErrorType.SERVER_ERROR:
            logger.warning(f"🌐 Discord server error: {error}")
        else:
            logger.error(f"❌ {error_type.value}: {error}")
    
    async def _handle_rate_limit(self, error: RateLimited):
        """Handle Discord rate limiting."""
        self.metrics.rate_limit_hits += 1
        
        retry_after = getattr(error, 'retry_after', 1.0)
        logger.warning(f"⏱️ Rate limited, waiting {retry_after:.2f}s")
        
        # Wait for the rate limit period
        await asyncio.sleep(retry_after)
    
    def _calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        if error_type == ErrorType.RATE_LIMIT:
            # Rate limit delays are handled separately
            return 0.0
        
        # Exponential backoff with jitter
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt)
        delay = min(delay, self.retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error handling metrics."""
        # Calculate error rate (errors per minute in last 5 minutes)
        now = datetime.now()
        recent_errors = [
            t for t in self.metrics.error_rate_window 
            if (now - t).total_seconds() <= 300
        ]
        error_rate = len(recent_errors) / 5.0  # Per minute
        
        metrics = {
            'total_errors': self.metrics.total_errors,
            'retry_attempts': self.metrics.retry_attempts,
            'rate_limit_hits': self.metrics.rate_limit_hits,
            'circuit_breaker_trips': self.metrics.circuit_breaker_trips,
            'error_rate_per_minute': error_rate,
            'errors_by_type': {k.value: v for k, v in self.metrics.errors_by_type.items()},
            'last_error_time': self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
            'circuit_breaker': self.circuit_breaker.get_stats()
        }
        
        return metrics
    
    def reset_metrics(self):
        """Reset error metrics."""
        self.metrics = ErrorMetrics()
        logger.info("📊 Error metrics reset")


# Decorators for easy error handling
def with_discord_retry(retry_config: Optional[RetryConfig] = None):
    """Decorator to add Discord error handling to async functions."""
    def decorator(func):
        error_handler = DiscordErrorHandler(retry_config)
        
        async def wrapper(*args, **kwargs):
            return await error_handler.handle_with_retry(func, *args, **kwargs)
        
        wrapper.error_handler = error_handler
        return wrapper
    return decorator


def with_circuit_breaker(circuit_config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to async functions."""
    def decorator(func):
        circuit_breaker = CircuitBreaker(circuit_config or CircuitBreakerConfig())
        
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = circuit_breaker
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler: Optional[DiscordErrorHandler] = None


def get_discord_error_handler() -> DiscordErrorHandler:
    """Get the global Discord error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = DiscordErrorHandler()
    return _global_error_handler


def init_discord_error_handler(
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None
) -> DiscordErrorHandler:
    """Initialize and return the Discord error handler."""
    global _global_error_handler
    _global_error_handler = DiscordErrorHandler(retry_config, circuit_config)
    return _global_error_handler