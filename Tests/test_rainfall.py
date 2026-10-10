import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rainfall_class import Region, CropRule

def test_rainfall_normal():
    region = Region("Test", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    assert region.annual_total() == 780
    assert region.wettest_month() == 11 # 0-indexed, 12th month is index 11
    assert region.driest_month() == 0

def test_croprule_normal():
    rule = CropRule("TestCrop", min_rain=50, max_rain=100)
    assert rule.classify_month(75) == "Good for TestCrop"
    assert rule.classify_month(20) == "Drought risk"
    assert rule.classify_month(150) == "Waterlogging risk"

def test_rainfall_edge_case_all_zeros():
    # Zero rainfall edge case
    region = Region("Zero", [0]*12)
    assert region.annual_total() == 0
    # CV of all zeros should ideally be handled without crashing, though np.std is 0
    cv = region.coefficient_of_variation()
    # It might return NaN if mean is 0 and it divides by 0, which is normal for numpy
    assert np.isnan(cv) or cv == 0
