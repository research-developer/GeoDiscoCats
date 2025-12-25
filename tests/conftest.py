"""
Pytest configuration and shared fixtures for partition geometry tests.
"""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def embedding_dim():
    """Default embedding dimension."""
    return 384
