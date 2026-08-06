"""
v6/security/authorize.py — Authorization chokepoint (RBAC seam)
=================================================================
A single authorize(principal, verb, target) chokepoint called at
every data-release point. Defaults to permit-all, writes audit line.

This cannot be retrofitted without a protocol break, so the seam
is built now even though policy is deferred.
"""

import time
import json
from typing import Optional


# Audit log entries
_audit_log = []


def authorize(
    principal: Optional[str],
    verb: str,
    target: str,
    building_id: str = "",
) -> bool:
    """
    Authorization chokepoint.

    Called at every data-release point in node/store.py and
    identify/gallery.py. Currently defaults to permit-all.

    Args:
        principal: requesting principal (None = anonymous/deferred)
        verb: action verb (e.g., "RESOLVE", "QUERY", "GOSSIP")
        target: target resource (e.g., "gallery:B3", "state:B1_P_042")
        building_id: local building ID for audit context

    Returns:
        True if authorized (currently always True)
    """
    entry = {
        "timestamp": time.time(),
        "building": building_id,
        "principal": principal,
        "verb": verb,
        "target": target,
        "decision": "PERMIT",  # default permit-all
    }
    _audit_log.append(entry)
    return True


def get_audit_log():
    """Return the accumulated audit log entries."""
    return list(_audit_log)


def clear_audit_log():
    """Clear the audit log (for testing)."""
    _audit_log.clear()


def audit(event_type: str, **kwargs):
    """
    Log a security-relevant event.

    Used for spoofing attempts, replay rejections, etc.
    """
    entry = {
        "timestamp": time.time(),
        "event_type": event_type,
    }
    entry.update(kwargs)
    _audit_log.append(entry)
