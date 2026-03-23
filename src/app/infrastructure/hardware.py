# src/app/infrastructure/hardware.py
"""
Hardware detection and device configuration.
Detects GPU availability using torch.
"""

import torch

from app.config import get_logger

logger = get_logger(__name__)

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using Device: {device}")
