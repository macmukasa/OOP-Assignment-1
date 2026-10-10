import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fish_class import FishStock, PriceModel, RiskAssessor

def test_fishstock_normal():
    stock = FishStock(r=0.4, K=10000, N0=4000)
    history = stock.simulate(weeks=52, h=0.1)
    assert len(history) == 53
    assert history[0] == 4000

def test_pricemodel_bounds():
    model = PriceModel(start_price=12000, bounds=(9000, 16000), seed=42)
    prices = model.simulate(weeks=52)
    assert len(prices) == 52
    assert all(9000 <= p <= 16000 for p in prices)

def test_fish_edge_case_zero_stock():
    # If starting stock is 0, it should remain 0
    stock = FishStock(r=0.4, K=10000, N0=0)
    history = stock.simulate(weeks=52, h=0.1)
    assert np.allclose(history, 0)
