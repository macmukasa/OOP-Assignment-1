import numpy as np
import statistics
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod


class Route:
    """
    Represents a taxi route with passenger counts and fare.
    """
    def __init__(self, name: str, passenger_counts: List[int], fare: float):
        if not passenger_counts:
            raise ValueError("Passenger counts cannot be empty.")
        if fare <= 0:
            raise ValueError("Fare must be positive.")
            
        self.name = name
        self.passenger_counts = np.array(passenger_counts, dtype=int)
        self.fare = float(fare)

    def daily_revenue(self) -> np.ndarray:
        """Returns the daily revenue."""
        return self.passenger_counts * self.fare

    def total_revenue(self) -> float:
        """Returns the total revenue over all days."""
        return np.sum(self.daily_revenue())

    def descriptive_statistics(self) -> Dict[str, float]:
        """Returns mean, variance, and standard deviation of passenger counts using statistics module."""
        counts = self.passenger_counts.tolist()
        if len(counts) < 2:
            raise ValueError("Need at least two data points for variance and standard deviation.")
        return {
            "mean": statistics.mean(counts),
            "variance": statistics.variance(counts),
            "std_dev": statistics.stdev(counts)
        }


class EquilibriumSolver:
    """
    Solves supply and demand linear systems.
    """
    @staticmethod
    def solve_equilibrium() -> Tuple[float, float]:
        """
        Solves for equilibrium price and quantity.
        Demand: Qd = 120 - 0.02P -> 0.02P + Qd = 120
        Supply: Qs = 10 + 0.03P  -> -0.03P + Qs = 10
        System:
        [ 0.02  1 ] [ P ] = [ 120 ]
        [-0.03  1 ] [ Q ] = [  10 ]
        """
        from scipy.linalg import solve
        
        coefficients = np.array([
            [0.02, 1],
            [-0.03, 1]
        ])
        constants = np.array([120, 10])
        
        solution = solve(coefficients, constants)
        price, quantity = solution[0], solution[1]
        
        return price, quantity


class Forecaster(ABC):
    """Abstract base class for forecasting models."""
    
    @abstractmethod
    def fit(self, data: np.ndarray):
        """Fits the forecaster on the data (if required)."""
        pass

    @abstractmethod
    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predicts future values for the given horizon."""
        pass
        
    def backtest(self, data: np.ndarray, start_idx: int) -> float:
        """
        Performs walk-forward backtesting.
        Predicts day t using data up to day t-1.
        Returns the Mean Absolute Error (MAE).
        """
        predictions = []
        actuals = data[start_idx:]
        
        for t in range(start_idx, len(data)):
            train_data = data[:t]
            self.fit(train_data)
            pred = self.predict(horizon=1)[0]
            predictions.append(pred)
            
        mae = np.mean(np.abs(np.array(predictions) - actuals))
        return mae


class MovingAverageForecaster(Forecaster):
    """3-day moving average forecaster."""
    def __init__(self, window: int = 3):
        self.window = window
        self.data = None
        
    def fit(self, data: np.ndarray):
        if len(data) < self.window:
            raise ValueError(f"Need at least {self.window} data points for MA forecaster.")
        self.data = data
        
    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.data is None:
            raise ValueError("Model must be fitted before predicting.")
        
        preds = []
        current_data = list(self.data)
        
        for _ in range(horizon):
            pred = np.mean(current_data[-self.window:])
            preds.append(pred)
            current_data.append(pred)
            
        return np.array(preds)


class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing forecaster."""
    def __init__(self, alpha: float):
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.last_level = None
        
    def fit(self, data: np.ndarray):
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        
        # Initialize level with the first data point
        level = data[0]
        for t in range(1, len(data)):
            level = self.alpha * data[t] + (1 - self.alpha) * level
            
        self.last_level = level
        
    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.last_level is None:
            raise ValueError("Model must be fitted before predicting.")
        # Simple exponential smoothing predicts a flat line (last level)
        return np.full(horizon, self.last_level)


class LinearTrendForecaster(Forecaster):
    """Linear trend forecaster using np.polyfit."""
    def __init__(self):
        self.coeffs = None
        self.last_t = None
        
    def fit(self, data: np.ndarray):
        if len(data) < 2:
            raise ValueError("Need at least two data points for linear trend.")
        
        t = np.arange(len(data))
        self.coeffs = np.polyfit(t, data, 1)
        self.last_t = len(data) - 1
        
    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.coeffs is None:
            raise ValueError("Model must be fitted before predicting.")
            
        t_future = np.arange(self.last_t + 1, self.last_t + 1 + horizon)
        preds = np.polyval(self.coeffs, t_future)
        return preds


class SeasonalNaiveForecaster(Forecaster):
    """Seasonal naive forecaster (predicts the value from one season ago)."""
    def __init__(self, seasonality: int = 7):
        self.seasonality = seasonality
        self.data = None
        
    def fit(self, data: np.ndarray):
        if len(data) < self.seasonality:
            raise ValueError(f"Need at least {self.seasonality} data points.")
        self.data = data
        
    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.data is None:
            raise ValueError("Model must be fitted before predicting.")
            
        preds = []
        for h in range(1, horizon + 1):
            idx = -self.seasonality + (h - 1) % self.seasonality
            preds.append(self.data[idx])
            
        return np.array(preds)

