"""
Notification Module
Handles HTTP webhook notifications with offline queue support.
"""

import json
import time
import threading
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from queue import Queue, Empty

# Import configuration
try:
    from src import config
except ImportError:
    import config


# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class FallNotification:
    """Data structure for a fall notification."""
    timestamp: str
    device_name: str
    device_location: str
    message: str
    fall_confidence: float
    event_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FallNotification':
        """Create from dictionary."""
        return cls(**data)


class NotificationQueue:
    """
    Persistent queue for notifications when offline.
    Uses a simple JSON file for storage.
    """
    
    def __init__(self, queue_file: Path, max_size: int = 100):
        self.queue_file = queue_file
        self.max_size = max_size
        self._lock = threading.Lock()
        
        # Ensure directory exists
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing queue
        self._queue: List[Dict[str, Any]] = self._load_queue()
    
    def _load_queue(self) -> List[Dict[str, Any]]:
        """Load queue from file."""
        try:
            if self.queue_file.exists():
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} queued notifications from file")
                    return data
        except Exception as e:
            logger.error(f"Error loading queue file: {e}")
        return []
    
    def _save_queue(self) -> None:
        """Save queue to file."""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump(self._queue, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving queue file: {e}")
    
    def add(self, notification: FallNotification) -> None:
        """Add notification to queue."""
        with self._lock:
            self._queue.append(notification.to_dict())
            
            # Trim if exceeds max size (remove oldest)
            while len(self._queue) > self.max_size:
                removed = self._queue.pop(0)
                logger.warning(f"Queue full, dropped oldest notification: {removed['event_id']}")
            
            self._save_queue()
            logger.info(f"Queued notification {notification.event_id}, queue size: {len(self._queue)}")
    
    def get_all(self) -> List[FallNotification]:
        """Get all notifications in queue."""
        with self._lock:
            return [FallNotification.from_dict(n) for n in self._queue]
    
    def remove(self, event_id: str) -> None:
        """Remove notification by event_id after successful send."""
        with self._lock:
            self._queue = [n for n in self._queue if n['event_id'] != event_id]
            self._save_queue()
    
    def clear(self) -> None:
        """Clear all notifications."""
        with self._lock:
            self._queue = []
            self._save_queue()
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)


class FallNotifier:
    """
    Handles sending fall detection notifications via HTTP webhook.
    Features:
    - Cooldown to prevent notification spam
    - Offline queue for when internet is unavailable
    - Automatic retry with backoff
    - Background queue processing
    """
    
    def __init__(
        self,
        webhook_url: str = None,
        device_name: str = None,
        device_location: str = None,
        cooldown_seconds: float = 30.0,
        timeout: int = 10,
        enable_queue: bool = True,
        queue_file: Path = None
    ):
        """
        Initialize the notifier.
        
        Args:
            webhook_url: URL to POST notifications to
            device_name: Name of this device
            device_location: Location of this device
            cooldown_seconds: Minimum seconds between notifications
            timeout: HTTP request timeout
            enable_queue: Whether to queue failed notifications
            queue_file: Path to queue file
        """
        self.webhook_url = webhook_url or config.WEBHOOK_URL
        self.device_name = device_name or config.DEVICE_NAME
        self.device_location = device_location or config.DEVICE_LOCATION
        self.cooldown_seconds = cooldown_seconds
        self.timeout = timeout
        self.enable_queue = enable_queue
        
        # Last notification timestamp for cooldown
        self._last_notification_time: float = 0
        self._lock = threading.Lock()
        
        # Notification counter for unique event IDs
        self._event_counter = 0
        
        # Offline queue
        if enable_queue:
            queue_path = queue_file or config.QUEUE_FILE
            self._queue = NotificationQueue(queue_path, config.MAX_QUEUE_SIZE)
            
            # Start background queue processor
            self._queue_processor_running = True
            self._queue_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
            self._queue_thread.start()
        else:
            self._queue = None
        
        logger.info(f"FallNotifier initialized: webhook={self.webhook_url}, device={self.device_name}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.device_name}-{timestamp}-{self._event_counter:04d}"
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active."""
        elapsed = time.time() - self._last_notification_time
        return elapsed < self.cooldown_seconds
    
    def _send_http_request(self, notification: FallNotification) -> bool:
        """
        Send HTTP POST request with notification data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare payload
            payload = json.dumps(notification.to_dict()).encode('utf-8')
            
            # Create request
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'VisioNull-FallDetector/1.0'
                },
                method='POST'
            )
            
            # Send request
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status = response.getcode()
                if 200 <= status < 300:
                    logger.info(f"Notification sent successfully: {notification.event_id}, status={status}")
                    return True
                else:
                    logger.warning(f"Notification failed with status {status}: {notification.event_id}")
                    return False
                    
        except urllib.error.URLError as e:
            logger.error(f"Network error sending notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    def notify_fall(self, confidence: float = 1.0) -> bool:
        """
        Send fall detection notification.
        
        Args:
            confidence: Fall detection confidence (0-1)
            
        Returns:
            True if notification was sent (or queued), False if cooldown active
        """
        with self._lock:
            # Check cooldown
            if self._is_cooldown_active():
                remaining = self.cooldown_seconds - (time.time() - self._last_notification_time)
                logger.debug(f"Cooldown active, {remaining:.1f}s remaining")
                return False
            
            # Update last notification time
            self._last_notification_time = time.time()
        
        # Create notification
        notification = FallNotification(
            timestamp=datetime.now().isoformat(),
            device_name=self.device_name,
            device_location=self.device_location,
            message="FALL DETECTED - Immediate attention required!",
            fall_confidence=round(confidence, 2),
            event_id=self._generate_event_id()
        )
        
        # Log the fall event
        self._log_fall_event(notification)
        
        # Try to send
        success = self._send_http_request(notification)
        
        if not success and self.enable_queue:
            # Queue for later
            self._queue.add(notification)
            logger.info(f"Notification queued for retry: {notification.event_id}")
            return True  # Queued counts as "handled"
        
        return success
    
    def _log_fall_event(self, notification: FallNotification) -> None:
        """Log fall event to file."""
        if not config.ENABLE_LOGGING:
            return
            
        try:
            log_file = config.FALL_LOG_FILE
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            log_entry = (
                f"[{notification.timestamp}] "
                f"FALL DETECTED | "
                f"Device: {notification.device_name} | "
                f"Location: {notification.device_location} | "
                f"Confidence: {notification.fall_confidence:.0%} | "
                f"Event ID: {notification.event_id}\n"
            )
            
            with open(log_file, 'a') as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.error(f"Error logging fall event: {e}")
    
    def _process_queue_loop(self) -> None:
        """Background loop to process queued notifications."""
        logger.info("Queue processor started")
        
        while self._queue_processor_running:
            try:
                time.sleep(config.QUEUE_RETRY_INTERVAL)
                self._process_queued_notifications()
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
    
    def _process_queued_notifications(self) -> None:
        """Process all queued notifications."""
        if not self._queue or self._queue.size() == 0:
            return
        
        logger.info(f"Processing {self._queue.size()} queued notifications")
        
        notifications = self._queue.get_all()
        
        for notification in notifications:
            success = self._send_http_request(notification)
            
            if success:
                self._queue.remove(notification.event_id)
            else:
                # Stop processing if we can't reach the server
                logger.warning("Failed to send queued notification, will retry later")
                break
    
    def get_queue_size(self) -> int:
        """Get number of queued notifications."""
        if self._queue:
            return self._queue.size()
        return 0
    
    def shutdown(self) -> None:
        """Shutdown the notifier gracefully."""
        self._queue_processor_running = False
        logger.info("FallNotifier shutdown")


# Convenience function for simple usage
_notifier_instance: Optional[FallNotifier] = None


def get_notifier() -> FallNotifier:
    """Get or create the global notifier instance."""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = FallNotifier()
    return _notifier_instance


def notify_fall(confidence: float = 1.0) -> bool:
    """Send fall notification using global notifier."""
    return get_notifier().notify_fall(confidence)
