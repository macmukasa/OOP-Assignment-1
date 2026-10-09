import numpy as np
import scipy.linalg
import scipy.optimize
import csv

class MicroGrid:
    """
    MicroGrid class representing a health centre micro-grid system.
    Solves for solar (x) and battery (y) usage given demands D1 and D2:
        3x + 2y = D1  (daytime load)
        4x +  y = D2  (critical-equipment load)
    """
    def __init__(self):
        # Coefficient matrix A
        self.A = np.array([[3.0, 2.0], 
                           [4.0, 1.0]])

    def check_well_posed(self):
        """
        Computes the determinant and condition number of the coefficient matrix.
        Returns:
            det (float): Determinant of the matrix.
            cond (float): Condition number of the matrix.
        """
        det = np.linalg.det(self.A)
        cond = np.linalg.cond(self.A)
        return det, cond

    def solve_day(self, d1, d2):
        """
        Solves the system for a single day using scipy.linalg.solve.
        """
        b = np.array([d1, d2])
        return scipy.linalg.solve(self.A, b)

    def solve_day_nnls(self, d1, d2):
        """
        Solves the system using Non-Negative Least Squares (NNLS) to handle 
        physically infeasible days (negative x or y).
        """
        b = np.array([d1, d2])
        solution, residual = scipy.optimize.nnls(self.A, b)
        return solution

    def solve_vectorized(self, D_matrix):
        """
        Solves the system for multiple days in a single vectorized call.
        Args:
            D_matrix: A 2xN numpy array of demands.
        Returns:
            A 2xN numpy array of solutions (x, y).
        """
        return scipy.linalg.solve(self.A, D_matrix)

    @staticmethod
    def get_interactive_demand():
        """
        Prompts the user interactively for D1 and D2 with robust validation.
        """
        def get_valid_input(prompt):
            while True:
                try:
                    val_str = input(prompt).strip()
                    if not val_str:
                        print("Input cannot be empty. Please try again.")
                        continue
                    val = float(val_str)
                    if val < 0:
                        print("Value cannot be negative. Please enter a positive number.")
                        continue
                    return val
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

        print("--- Interactive Mode ---")
        d1 = get_valid_input("Enter Daytime Load (D1) in kWh: ")
        d2 = get_valid_input("Enter Critical-Equipment Load (D2) in kWh: ")
        return d1, d2

    @staticmethod
    def generate_csv_data(filename="demand_data.csv", days=30, seed=42):
        """
        Generates 30 days of demand data with a weekly pattern and noise,
        saving it to a CSV file. Includes logic to occasionally force infeasible 
        demands to test edge-cases.
        """
        rng = np.random.default_rng(seed)
        
        base_d1 = 120.0
        base_d2 = 90.0
        
        data = []
        for day in range(days):
            # Weekly pattern: slightly higher demand on weekdays (day % 7 < 5)
            # and lower on weekends.
            is_weekend = (day % 7) >= 5
            multiplier = 0.8 if is_weekend else 1.1
            
            if day == 10:
                # Edge case: D1 > 2*D2 forces negative x
                d1 = 180.0
                d2 = 60.0
            elif day == 20:
                # Edge case: D1 < 0.75*D2 forces negative y
                d1 = 60.0
                d2 = 130.0
            else:
                noise_d1 = rng.normal(0, 10)
                noise_d2 = rng.normal(0, 8)
                d1 = max(0, base_d1 * multiplier + noise_d1)
                d2 = max(0, base_d2 * multiplier + noise_d2)
                
            data.append([day+1, round(d1, 2), round(d2, 2)])
            
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Day", "D1", "D2"])
            writer.writerows(data)
            
        return filename

    @staticmethod
    def load_csv_data(filename="demand_data.csv"):
        """
        Loads demand data from a CSV file.
        Returns:
            A 2xN numpy array of demands.
        """
        d1_list = []
        d2_list = []
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                d1_list.append(float(row["D1"]))
                d2_list.append(float(row["D2"]))
        return np.array([d1_list, d2_list])


class HybridMicroGrid(MicroGrid):
    """
    Extension: Hybrid micro-grid adding a diesel generator (z).
    New constraint: e.g., x + y + z = D3 (total energy produced).
    """
    def __init__(self, new_equation=None):
        super().__init__()
        # Default 3rd equation: 1x + 1y + 1z = D3
        if new_equation is None:
            new_equation = [1.0, 1.0, 1.0]
            
        self.A_hybrid = np.array([
            [3.0, 2.0, 0.0],
            [4.0, 1.0, 0.0],
            new_equation
        ])

    def check_well_posed_hybrid(self):
        det = np.linalg.det(self.A_hybrid)
        cond = np.linalg.cond(self.A_hybrid)
        return det, cond

    def solve_day_hybrid(self, d1, d2, d3):
        b = np.array([d1, d2, d3])
        return scipy.linalg.solve(self.A_hybrid, b)

