"""
Notification sub-package — HTTP webhook notifications with offline queue support.
"""

from src.notification.notifier import (
    FallNotifier,
    FallNotification,
    NotificationQueue,
    get_notifier,
    notify_fall,
)
