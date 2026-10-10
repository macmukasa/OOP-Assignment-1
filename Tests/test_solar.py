import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.solar_class import MicroGrid, HybridMicroGrid
from scipy.linalg import LinAlgError

def test_solar_normal_solve():
    grid = MicroGrid()
    sol = grid.solve_day(120, 10)
    # 3x + 2y = 120, 4x + y = 10
    # x = -20, y = 90
    assert np.isclose(sol[0], -20)
    assert np.isclose(sol[1], 90)

def test_solar_nnls_fallback():
    grid = MicroGrid()
    sol = grid.solve_day_nnls(120, 10)
    assert sol[0] >= 0
    assert sol[1] >= 0

def test_solar_edge_case_singular():
    # A dependent third equation
    hybrid = HybridMicroGrid(new_equation=[6.0, 4.0, 0.0]) # 2 * first equation
    with pytest.raises(LinAlgError):
        hybrid.solve_day_hybrid(120, 90, 240)
