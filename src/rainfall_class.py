import numpy as np
import scipy.spatial.distance as dist
from scipy.signal import find_peaks

class Region:
    """
    Holds rainfall data for a specific region and provides descriptive statistical methods.
    """
    def __init__(self, name, rainfall):
        self.name = name
        self.rainfall = np.array(rainfall, dtype=float)
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def annual_total(self):
        """Returns the total annual rainfall."""
        return np.sum(self.rainfall)

    def mean_monthly(self):
        """Returns the mean monthly rainfall."""
        return np.mean(self.rainfall)

    def wettest_month(self):
        """Returns the name and rainfall amount of the wettest month."""
        idx = np.argmax(self.rainfall)
        return self.months[idx], self.rainfall[idx]

    def driest_month(self):
        """Returns the name and rainfall amount of the driest month."""
        idx = np.argmin(self.rainfall)
        return self.months[idx], self.rainfall[idx]

    def coefficient_of_variation(self):
        """Computes the coefficient of variation (std dev / mean)."""
        if self.mean_monthly() == 0:
            return 0.0
        return np.std(self.rainfall, ddof=1) / self.mean_monthly()

class CropRule:
    """
    Defines suitability rules for a crop based on monthly rainfall ranges.
    """
    def __init__(self, crop_name, min_rain, max_rain, source):
        self.crop_name = crop_name
        self.min_rain = min_rain
        self.max_rain = max_rain
        self.source = source

    def classify_month(self, rainfall_mm):
        """Classifies a specific month's rainfall for this crop."""
        if rainfall_mm < self.min_rain:
            return "Drought risk"
        elif rainfall_mm > self.max_rain:
            return "Waterlogging risk"
        else:
            return f"Good for {self.crop_name}"
            
    def classify_numeric(self, rainfall_mm):
        """Returns numeric classification for heatmaps: 0=Drought, 1=Good, 2=Waterlogging"""
        if rainfall_mm < self.min_rain:
            return 0
        elif rainfall_mm > self.max_rain:
            return 2
        else:
            return 1

class RainfallAnalyzer:
    """
    Provides analytical tools to compare multiple Region objects.
    """
    @staticmethod
    def cosine_similarity(v1, v2):
        """
        Computes cosine similarity of two vectors properly.
        Formula: dot(v1, v2) / (norm(v1) * norm(v2))
        """
        dot_prod = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_prod / (norm_v1 * norm_v2)

    @staticmethod
    def verify_cosine(v1, v2):
        """Verifies custom cosine similarity against scipy.spatial.distance.cosine."""
        custom_sim = RainfallAnalyzer.cosine_similarity(v1, v2)
        # scipy calculates cosine DISTANCE, so similarity is 1 - distance
        scipy_sim = 1.0 - dist.cosine(v1, v2)
        return np.isclose(custom_sim, scipy_sim)

    @staticmethod
    def detect_seasons(rainfall):
        """
        Detects rainy seasons using scipy.signal.find_peaks.
        Returns the modality string ('Unimodal', 'Bimodal', etc.) and peak indices.
        """
        peaks, _ = find_peaks(rainfall, prominence=15, distance=2)
        
        n_peaks = len(peaks)
        if n_peaks == 1:
            return "Unimodal", peaks
        elif n_peaks == 2:
            return "Bimodal", peaks
        else:
            return f"{n_peaks}-modal", peaks

