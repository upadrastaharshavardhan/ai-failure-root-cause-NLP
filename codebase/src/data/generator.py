"""
Synthetic failure data generator.
Produces realistic multi-service stack traces and error messages
mapped to root-cause categories for training and demos.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT_CAUSE_TEMPLATES: Dict[str, List[str]] = {
    "NullPointer": [
        'NullPointerException: Cannot invoke "String.length()" because "userId" is null',
        "java.lang.NullPointerException at com.example.AuthService.validateToken(AuthService.java:87)",
        "NPE when accessing user profile because session object was null",
        "NullPointerException: Cannot invoke \"User.getEmail()\" because \"user\" is null",
        "java.lang.NullPointerException: text is null at String.concat",
    ],
    "Configuration": [
        "IllegalStateException: Property 'db.url' is not set in application.yml",
        "Failed to start: missing required environment variable JWT_SECRET",
        "Config error: invalid value for max.pool.size = -1",
        "BeanCreationException: Error creating bean with name 'dataSource': Property 'url' is required",
        "IllegalArgumentException: Invalid configuration value for redis.host",
    ],
    "Dependency": [
        "ConnectException: Connection refused to payment-gateway:8080",
        "FeignException: status 503 from downstream service inventory-service",
        "TimeoutException while calling external SMS provider",
        "HttpServerErrorException: 502 Bad Gateway from recommendation-service",
        "ResourceAccessException: I/O error on GET request for \"http://pricing-service\"",
    ],
    "ResourceExhaustion": [
        "OutOfMemoryError: Java heap space",
        "CannotGetJdbcConnectionException: Connection pool exhausted",
        "Too many open files - ulimit exceeded",
        "java.lang.OutOfMemoryError: GC overhead limit exceeded",
        "RejectedExecutionException: Task rejected from ThreadPoolExecutor",
    ],
    "RaceCondition": [
        "ConcurrentModificationException in shopping cart update",
        "OptimisticLockException: version mismatch on order entity",
        "Deadlock found when trying to get lock; try restarting transaction",
        "IllegalStateException: Concurrent access to shared resource detected",
        "StaleObjectStateException: Row was updated or deleted by another transaction",
    ],
    "Auth": [
        "AccessDeniedException: User does not have role ADMIN",
        "JwtException: Token has expired",
        "AuthenticationException: Bad credentials",
        "InsufficientAuthenticationException: Full authentication is required",
        "InvalidBearerTokenException: Invalid token signature",
    ],
    "Validation": [
        "MethodArgumentNotValidException: email must be a well-formed email address",
        "ConstraintViolationException: age must be greater than 0",
        "IllegalArgumentException: order quantity cannot be negative",
        "BindException: Failed to bind request body (missing required field 'customerId')",
        "HttpMessageNotReadableException: JSON parse error: Unexpected character",
    ],
    "Network": [
        "SocketTimeoutException: Read timed out after 30000 ms",
        "ConnectTimeoutException: connect timed out",
        "UnknownHostException: inventory.internal.svc",
        "NoRouteToHostException: No route to host",
        "SSLHandshakeException: Remote host terminated the handshake",
    ],
    "Database": [
        "SQLException: Duplicate entry 'ORD-99821' for key 'orders.PRIMARY'",
        "QueryTimeoutException: query execution exceeded 30s",
        "DataIntegrityViolationException: foreign key constraint fails",
        "SQLTransientConnectionException: Connection is not available",
        "PessimisticLockingFailureException: could not obtain lock on row",
    ],
}

SERVICES = [
    "auth-service",
    "order-service",
    "payment-service",
    "inventory-service",
    "user-service",
    "notification-service",
    "api-gateway",
    "pricing-service",
    "recommendation-service",
]


def _generate_stacktrace(service: str, error_line: str) -> str:
    svc_class = "".join(w.capitalize() for w in service.replace("-", " ").split())
    return (
        f'Exception in thread "http-nio-8080-exec-{random.randint(1, 20)}"\n'
        f"{error_line}\n"
        f"\tat com.example.{svc_class}.Controller.handle(Controller.java:{random.randint(20, 150)})\n"
        f"\tat org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1072)\n"
        f"\tat org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:149)\n"
        f"\tat java.base/java.lang.Thread.run(Thread.java:833)"
    )


def generate_dataset(
    n_samples: int = 3000,
    seed: int = 42,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic failure dataset.

    Parameters
    ----------
    n_samples : int
        Number of failure records to create.
    seed : int
        Random seed for reproducibility.
    categories : list, optional
        Subset of root-cause categories. Defaults to all keys in ROOT_CAUSE_TEMPLATES.

    Returns
    -------
    pd.DataFrame
        Columns: failure_id, timestamp, service, error_message, stacktrace,
                 full_text, root_cause_category
    """
    random.seed(seed)
    np.random.seed(seed)

    if categories is None:
        categories = list(ROOT_CAUSE_TEMPLATES.keys())

    records = []
    for i in range(n_samples):
        category = random.choice(categories)
        error_msg = random.choice(ROOT_CAUSE_TEMPLATES[category])
        service = random.choice(SERVICES)

        # Inject realistic noise
        if random.random() < 0.35:
            error_msg = f"{error_msg} | requestId={random.randint(100000, 999999)}"
        if random.random() < 0.20:
            error_msg = f"{error_msg} | correlationId={random.randbytes(8).hex()}"

        stack = _generate_stacktrace(service, error_msg)
        full_text = f"Service: {service}\nError: {error_msg}\nStacktrace:\n{stack}"

        records.append(
            {
                "failure_id": f"F-{i + 1:05d}",
                "timestamp": datetime.now() - timedelta(hours=random.randint(1, 2000)),
                "service": service,
                "error_message": error_msg,
                "stacktrace": stack,
                "full_text": full_text,
                "root_cause_category": category,
            }
        )

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset(100)
    print(df.head())
    print("\nCategory distribution:")
    print(df["root_cause_category"].value_counts())
