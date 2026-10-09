import numpy as np

class FishStock:
    """
    Simulates fish stock using a discrete logistic growth model with harvesting.
    Formula: N(t+1) = N(t) + r * N(t) * (1 - N(t)/K) - h * N(t)
    """
    def __init__(self, r=0.4, K=10000.0, N0=4000.0):
        self.r = r
        self.K = K
        self.N0 = N0

    def simulate(self, weeks=52, h=0.1, closed_season_weeks=None):
        """
        Simulates the fish stock over a given number of weeks.
        
        Args:
            weeks (int): Number of weeks to simulate.
            h (float): Base harvest rate.
            closed_season_weeks (set or list): specific weeks (0-indexed) where harvest is 0.
            
        Returns:
            stock (ndarray): Array of stock values for weeks 0 to 'weeks'.
            harvests (ndarray): Array of harvest amounts (in tonnes) for weeks 0 to 'weeks'-1.
        """
        stock = np.zeros(weeks + 1)
        stock[0] = self.N0
        harvests = np.zeros(weeks)
        closed_season_weeks = set(closed_season_weeks) if closed_season_weeks else set()

        for t in range(weeks):
            # Enforce closed season by dropping harvest rate to 0
            current_h = 0.0 if t in closed_season_weeks else h
            
            # Harvest for the week
            harvests[t] = current_h * stock[t]
            
            # Calculate next week's stock using the discrete logistic equation
            next_stock = stock[t] + self.r * stock[t] * (1.0 - stock[t] / self.K) - harvests[t]
            
            # Biologically, stock cannot drop below 0
            stock[t+1] = max(0.0, next_stock)

        return stock, harvests

class PriceModel:
    """
    Simulates the weekly price of fish in UGX/kg as a bounded random walk.
    """
    def __init__(self, start_price=12000.0, min_price=9000.0, max_price=16000.0, volatility=500.0):
        self.start_price = start_price
        self.min_price = min_price
        self.max_price = max_price
        self.volatility = volatility

    def simulate(self, weeks=52, seed=None):
        """
        Generates a simulated price path over a given number of weeks.
        """
        rng = np.random.default_rng(seed)
        prices = np.zeros(weeks)
        current_price = self.start_price
        
        for t in range(weeks):
            prices[t] = current_price
            step = rng.normal(0, self.volatility)
            current_price += step
            # Clip the price to stay within defined economic bounds
            current_price = np.clip(current_price, self.min_price, self.max_price)
            
        return prices

class RiskAssessor:
    """
    Classifies risk based on the Coefficient of Variation (CV) and calculates Value-at-Risk.
    """
    def __init__(self, low_threshold=0.10, high_threshold=0.25):
        # Rule: CV < 10% is Low Risk, 10-25% is Medium Risk, > 25% is High Risk.
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def classify_risk(self, cv):
        """
        Returns a risk classification based on the supplied CV.
        """
        if cv < self.low_threshold:
            return "Low Risk"
        elif cv < self.high_threshold:
            return "Medium Risk"
        else:
            return "High Risk"
            
    def calculate_var(self, simulated_revenues, percentile=5):
        """
        Calculates the Value-at-Risk (VaR) from an array of simulated revenues.
        In this business context, we return the absolute value at the given percentile 
        (e.g., we are 95% confident revenue will exceed this VaR value).
        """
        return np.percentile(simulated_revenues, percentile)

