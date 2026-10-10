import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.taxi_class import Route, MovingAverageForecaster

def test_route_normal():
    r = Route("Test", [10, 20, 30], 2000)
    assert r.total_revenue() == 120000 # 60 * 2000
    stats = r.descriptive_statistics()
    assert stats['mean'] == 20

def test_forecaster_normal():
    data = np.array([10, 20, 30])
    model = MovingAverageForecaster(window=2)
    model.fit(data)
    pred = model.predict(horizon=1)
    assert np.isclose(pred[0], 25) # (20+30)/2

def test_taxi_edge_case_invalid_inputs():
    with pytest.raises(ValueError):
        Route("Test", [], 2000) # empty passenger counts
    
    with pytest.raises(ValueError):
        Route("Test", [10, 20], -500) # negative fare
        
    model = MovingAverageForecaster(window=2)
    with pytest.raises(ValueError):
        model.predict(horizon=1) # predicting before fitting
