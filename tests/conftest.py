import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg():
    from core.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    """Deterministic synthetic fixture generated fresh for the test session (no network, no GPU)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_demo_fixture import make
    d = tmp_path_factory.mktemp("demo")
    make(d)
    return d


@pytest.fixture
def plane():
    """A tilted plane z = 0.1 x (10 % grade, 5.71 deg) on a 40 m grid."""
    x = np.arange(50) * 40.0
    return np.tile(0.1 * x, (40, 1)).astype(np.float32)
