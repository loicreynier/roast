"""Statistics functions tests."""


import numpy as np
import pytest

import roast.stats


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_rms(n: int) -> None:
    """Test RMS of sine wave."""
    e = 1.0 / n  # Floating precision
    t = np.linspace(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * t)
    rms = 1.0 / np.sqrt(2.0)
    rms_n = roast.stats.rms(y)
    assert np.abs(rms_n - rms) < e
