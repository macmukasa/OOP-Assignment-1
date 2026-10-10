import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ubos_class import DistrictPopulation, LinearForecaster

def test_ubos_normal():
    dp = DistrictPopulation([2020, 2021], [1000, 1100])
    assert len(dp) == 2
    cagr = dp.cagr()
    assert np.isclose(cagr, 0.1)

def test_ubos_statistics():
    dp = DistrictPopulation([2020, 2021, 2022], [1000, 1100, 1200])
    stats = dp.compute_statistics()
    assert stats['mean_stat'] == 1100
    assert stats['variance_stat'] == 10000

def test_ubos_edge_case_negative():
    with pytest.raises(ValueError):
        DistrictPopulation([2020, 2021], [1000, -100])

def test_ubos_edge_case_length_mismatch():
    with pytest.raises(ValueError):
        DistrictPopulation([2020, 2021], [1000])
