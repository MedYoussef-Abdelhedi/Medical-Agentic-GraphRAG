"""Run the automated test suite using the proper source-package path."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CORE = SRC / "medical_graphrag"
TESTS = ROOT / "tests"
if not CORE.is_dir() or not TESTS.is_dir():
    raise SystemExit("src/medical_graphrag or tests directory not found")
sys.path.insert(0, str(SRC))
os.chdir(CORE)  # Preserve legacy working-directory semantics for local fixtures.
suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
