# OOP with Python - Assignment 1

This repository contains the solutions for the MSCS & MSDS OOP with Python Assignment (Advent 2026). The project tackles five Ugandan context-based mini-projects using object-oriented programming (OOP), mathematical modeling, and data visualization.

## Repository Structure

- `*.ipynb`: The main Jupyter Notebooks containing the executed code, visualizations, and analytical summaries for each mini-project.
- `src/`: Contains the reusable Python modules (`_class.py`) defining the domain classes for each project.
- `Tests/`: Contains the `pytest` unit test files covering normal and edge cases for all modules.
- `requirements.txt`: The Python dependencies required to run the notebooks.

## Setup and Execution

To set up the project locally and run the notebooks:

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
   ```

2. **Activate the virtual environment (or create a new one):**
   ```bash
   # Create environment (if not already existing)
   python -m venv oop
   
   # Activate environment (Windows)
   oop\Scripts\activate
   
   # Activate environment (Mac/Linux)
   source oop/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the tests (Optional):**
   ```bash
   python -m pytest Tests
   ```

5. **Run the Notebooks:**
   Launch Jupyter Notebook to view and execute the `.ipynb` files. The notebooks are configured for reproducibility; you can safely click **Restart & Run All**.
   ```bash
   jupyter notebook
   ```

## Summary of Findings

1. **Mini-Project 1: UBOS District Population Forecaster (`ubos_impl.ipynb`)**  
   All analyzed districts are experiencing steady population growth. Linear and exponential models proved significantly more reliable than the Fibonacci-ratio approach. Accurate forecasting translates directly to actionable planning, ensuring sufficient primary school classrooms can be built to accommodate the population by 2029.

2. **Mini-Project 2: Solar Micro-Grid Dispatch Planner (`solar_impl.ipynb`)**  
   The micro-grid dispatch can be effectively solved as a system of linear equations. Tracking the condition number confirmed the system was well-conditioned against demand fluctuations. When physical constraints were broken (i.e., requiring negative energy), switching to a Non-Negative Least Squares (NNLS) solver gracefully prevented unphysical dispatches.

3. **Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model (`fish_impl.ipynb`)**  
   The discrete logistic growth model provides a biologically realistic framework compared to unbounded Fibonacci bounds. Furthermore, valuing revenue risk using Monte Carlo simulations to calculate a 5% Value-at-Risk (VaR) proved far superior and more statistically robust than relying on an arbitrary raw variance threshold. 

4. **Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser (`rainfall_impl.ipynb`)**  
   Applying distance measures (Cosine Similarity, Pearson, Euclidean) alongside automated peak detection algorithms effectively mapped out unimodal and bimodal rainfall seasons. These results align perfectly with Uganda's established climate zones and allow for robust, data-driven agricultural crop advisories.

5. **Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner (`taxi_route_impl.ipynb`)**  
   Supply-demand equilibrium analysis revealed that current fares on some routes are undervalued, leading to excess demand. While simple time-series models handle flat trends, a Seasonal Naïve approach effectively captures the strong weekly demand patterns (e.g., Friday spikes), allowing for optimized fleet deployment with a sensible buffer.

## AI Declaration

AI Coding Assistants were utilized during the completion of this assignment. The AI was used as a pair-programmer to assist in structuring the Object-Oriented code, and generating the boilerplate for `pytest` unit tests. All mathematical modeling, architectural choices, and analytical interpretations were actively guided and reviewed to strictly adhere to the assignment's rubric.

