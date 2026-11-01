"""Lance les tests de la base avec les anciens imports conserves."""
from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "src" / "medical_graphrag"
TESTS = ROOT / "tests"

if not CORE.is_dir() or not TESTS.is_dir():
    raise SystemExit("Migration non terminee : src/medical_graphrag ou tests absent")

# Les scripts historiques ont des imports absolus de modules voisins.
sys.path.insert(0, str(CORE))
os.chdir(CORE)
suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
