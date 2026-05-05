"""
Authentication state manager for Meridian customer sessions.

Each Gradio session gets its own AuthSession instance so multiple
concurrent users don't share state.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuthSession:
    """
    Holds authentication state for one chat session.
    Reset on logout or new session start.
    """
    authenticated: bool = False
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None
    customer_id: Optional[str] = None

    # Track how many failed attempts this session to rate-limit brute force
    failed_attempts: int = 0
    MAX_ATTEMPTS: int = field(default=5, init=False, repr=False)

    def login(self, email: str, customer_data: dict) -> None:
        """Mark session as authenticated and store customer info."""
        self.authenticated = True
        self.customer_email = email
        self.customer_name = customer_data.get("name", email)
        self.customer_id = customer_data.get("id") or customer_data.get("customer_id")
        self.failed_attempts = 0

    def logout(self) -> None:
        """Reset session to unauthenticated state."""
        self.authenticated = False
        self.customer_email = None
        self.customer_name = None
        self.customer_id = None
        self.failed_attempts = 0

    def record_failed_attempt(self) -> None:
        self.failed_attempts += 1

    @property
    def is_locked_out(self) -> bool:
        return self.failed_attempts >= self.MAX_ATTEMPTS

    def greeting_name(self) -> str:
        return self.customer_name or self.customer_email or "there"
