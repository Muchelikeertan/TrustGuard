"""TrustGuard multi-modal verification package."""
from trustguard.chat_check import analyze_chat, Signal
from trustguard.consistency_engine import (
    evaluate_consistency,
    MultimodalConsistencyResult,
    ContradictionMap,
)

__all__ = [
    "analyze_chat",
    "Signal",
    "evaluate_consistency",
    "MultimodalConsistencyResult",
    "ContradictionMap",
]
