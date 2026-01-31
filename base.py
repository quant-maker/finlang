"""
Base classes for trading signal generation.

This module provides the foundational abstractions:
- Signal: A trading signal with position, confidence, and metadata
- BaseStrategy: Abstract base class for all trading strategies
- StrategyType: Enum for categorizing strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, Literal
import json


class StrategyType(Enum):
    """Strategy category for classification."""
    VOTING = "voting"
    MLP = "mlp"
    DYNAMIC = "dynamic"


ConfidenceLevel = Literal["low", "medium", "high", "error"]
TrendState = Literal["bearish", "neutral", "bullish"]


@dataclass
class Signal:
    """
    Trading signal with position recommendation and metadata.
    
    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        position: Target position from -1.0 (full short) to 1.0 (full long)
        confidence: Signal confidence level
        current_price: Current market price
        score: Signal strength from -100 to +100
        factors: Dictionary of factors that influenced the signal
        trend_state: Overall market trend assessment
        reasoning: Human-readable explanation of the signal
        strategy_used: Name of the strategy that generated this signal
        timestamp: When the signal was generated (UTC)
        prediction_time: Timestamp of the latest prediction used
        data_points: Number of prediction data points used
    """
    symbol: str
    position: float
    confidence: ConfidenceLevel
    current_price: float = 0.0
    score: float = 0.0
    factors: Dict[str, Any] = field(default_factory=dict)
    trend_state: TrendState = "neutral"
    reasoning: str = ""
    analysis_type: str = "kronos"
    strategy_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prediction_time: str = ""
    data_points: int = 0
    
    def __post_init__(self):
        """Validate signal values."""
        if not -1.0 <= self.position <= 1.0:
            raise ValueError(f"Position must be between -1.0 and 1.0, got {self.position}")
        if self.confidence not in ("low", "medium", "high", "error"):
            raise ValueError(f"Invalid confidence level: {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert signal to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, path: str) -> None:
        """Save signal to JSON file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> "Signal":
        """Load signal from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def error(cls, symbol: str, error_message: str) -> "Signal":
        """Create an error signal."""
        return cls(
            symbol=symbol,
            position=0.0,
            confidence="error",
            current_price=0.0,
            score=0.0,
            reasoning=f"[ERROR] {error_message}",
            strategy_used="error",
        )
    
    @property
    def action(self) -> str:
        """Get action string based on position."""
        if self.position > 0:
            return "LONG"
        elif self.position < 0:
            return "SHORT"
        return "FLAT"
    
    @property
    def is_valid(self) -> bool:
        """Check if signal is valid (not an error)."""
        return self.confidence != "error"


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Subclasses must implement:
    - name: Property returning strategy name
    - strategy_type: Property returning StrategyType
    - generate_signal: Method to generate trading signal
    
    Optional overrides:
    - description: Human-readable description
    - config: Dictionary of strategy parameters
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for identification."""
        pass
    
    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Strategy type for categorization."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description of the strategy."""
        return f"{self.name} strategy"
    
    @property
    def config(self) -> Dict[str, Any]:
        """Strategy configuration parameters."""
        return {}
    
    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        predictions: Dict[str, Dict[str, float]],
        prices: Any,
        **kwargs
    ) -> Signal:
        """
        Generate a trading signal.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            predictions: Dict mapping timestamps to prediction data
                Each prediction should contain:
                - sig_24h: 24h upside probability (0-1)
                - sig_2h: 2h upside probability (0-1)
                - current_close: Current close price
            prices: Price DataFrame with OHLCV data
            **kwargs: Additional strategy-specific arguments
        
        Returns:
            Signal object with trading recommendation
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
