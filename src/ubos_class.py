import numpy as np
import statistics as st
from abc import ABC, abstractmethod

class DistrictPopulation:
    """
    Class that stores years and populations as NumPy arrays, 
    validates the input (equal lengths, no negative values) 
    and implements __repr__ and __len__.
    """
    def __init__(self, years, populations):
        self.years = np.array(years)
        self.populations = np.array(populations)

        if len(self.years) != len(self.populations):
            raise ValueError("Years and populations must have the same length.")
        if np.any(self.populations < 0):
            raise ValueError("Populations cannot contain negative values.")

    def __repr__(self):
        return f"DistrictPopulation(years={self.years}, populations={self.populations})"

    def __len__(self):
        return len(self.years)

    def compute_statistics(self):
        """
        Compute the mean, median, variance, and standard deviation for the populations
        using both the statistics module and numpy.
        """
        pop_list = self.populations.tolist()
        mean_stat = st.mean(pop_list)
        median_stat = st.median(pop_list)
        variance_stat = st.variance(pop_list)
        std_dev_stat = st.stdev(pop_list)

        mean_np = np.mean(self.populations)
        median_np = np.median(self.populations)
        variance_np = np.var(self.populations, ddof=1)  # Sample variance
        std_dev_np = np.std(self.populations, ddof=1)  # Sample standard deviation

        return {
            'mean_stat': mean_stat, 'median_stat': median_stat,
            'variance_stat': variance_stat, 'std_dev_stat': std_dev_stat,
            'mean_np': mean_np, 'median_np': median_np,
            'variance_np': variance_np, 'std_dev_np': std_dev_np
        }

    def yoy_growth_rates(self):
        """Compute year-on-year growth rates."""
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self):
        """Compute the Compound Annual Growth Rate (CAGR)."""
        n = len(self.populations) - 1
        if n > 0:
            return (self.populations[-1] / self.populations[0]) ** (1 / n) - 1
        return 0.0


class Forecaster(ABC):
    @abstractmethod
    def fit(self, years, populations):
        pass

    @abstractmethod
    def predict(self, horizon):
        pass
    
    @abstractmethod
    def fitted_values(self):
        pass


class LinearForecaster(Forecaster):
    def __init__(self):
        self.coeffs = None
        self.last_year = None
        self.fit_years = None
        self.base_year = None

    def fit(self, years, populations):
        self.base_year = years[0]
        x = years - self.base_year
        self.coeffs = np.polyfit(x, populations, 1)
        self.last_year = years[-1]
        self.fit_years = years
        
    def predict(self, horizon):
        future_years = np.arange(self.last_year + 1, self.last_year + 1 + horizon)
        x_future = future_years - self.base_year
        return np.polyval(self.coeffs, x_future)
        
    def fitted_values(self):
        x = self.fit_years - self.base_year
        return np.polyval(self.coeffs, x)


class ExponentialForecaster(Forecaster):
    def __init__(self):
        self.last_year = None
        self.last_pop = None
        self.first_pop = None
        self.cagr = None
        self.fit_years = None

    def fit(self, years, populations):
        self.last_year = years[-1]
        self.last_pop = populations[-1]
        self.first_pop = populations[0]
        n = len(populations) - 1
        self.cagr = (populations[-1] / populations[0]) ** (1 / n) - 1 if n > 0 else 0
        self.fit_years = years

    def predict(self, horizon):
        k = np.arange(1, horizon + 1)
        return self.last_pop * ((1 + self.cagr) ** k)

    def fitted_values(self):
        k = np.arange(len(self.fit_years))
        return self.first_pop * ((1 + self.cagr) ** k)


class FibonacciForecaster(Forecaster):
    def __init__(self):
        self.last_year = None
        self.last_pop = None
        self.fit_years = None
        self.populations = None
        self.fib = [1, 1]

    def fit(self, years, populations):
        self.last_year = years[-1]
        self.last_pop = populations[-1]
        self.fit_years = years
        self.populations = populations
        self.fib = [1, 1]

    def _get_fib(self, n):
        while len(self.fib) <= n:
            self.fib.append(self.fib[-1] + self.fib[-2])
        return self.fib[n]

    def predict(self, horizon):
        predictions = []
        current_pop = self.last_pop
        for i in range(1, horizon + 1):
            ratio = self._get_fib(i + 1) / self._get_fib(i)
            current_pop = current_pop * ratio
            predictions.append(current_pop)
        return np.array(predictions)
        
    def fitted_values(self):
        # We simulate the fit by applying the Fibonacci ratios starting from the first year.
        fitted = [self.populations[0]]
        for i in range(1, len(self.fit_years)):
            ratio = self._get_fib(i + 1) / self._get_fib(i)
            fitted.append(fitted[-1] * ratio)
        return np.array(fitted)

