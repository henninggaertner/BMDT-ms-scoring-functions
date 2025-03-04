import numpy as np
import pandas as pd
import time;
from functools import wraps

def simple_scoring_function(theoretical_spectrum: np.ndarray, experimental_spectrum: np.ndarray,
                            t: int = 20) -> float:
    """
    parameters:
    theoretical_spectrum: np.ndarray
        Theoretical spectrum to compare against the experimental spectrum
    experimental_spectrum: np.ndarray
        Experimental spectrum to compare against the theoretical spectrum
    t: int
        Mass tolerance in parts per million (ppm)
    ----------
    returns:
    float
        The score of the theoretical spectrum against the experimental spectrum.
        The score is calculated as the number of matched peaks divided by the total number of peaks in the theoretical
        spectrum.
    """

    # Ensure theoretical_spectrum is one-dimensional (m/z values only)
    if theoretical_spectrum.ndim != 1:
        raise ValueError("Theoretical spectrum must be a 1D array containing only m/z values.")
    # Extract only m/z values from the experimental spectrum since we don't need intensity values for this simple
    # scoring method
    experimental_spectrum = experimental_spectrum[:, 0]
    matches = []
    for peak in theoretical_spectrum:
        tolerance = peak * t / 1e6
        matched = np.where(np.abs(experimental_spectrum - peak) <= tolerance * peak)
        if len(matched[0]) > 0:
            matches.append(peak)
            experimental_spectrum = np.delete(experimental_spectrum, matched)
    return len(matches) / len(theoretical_spectrum)


def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate elapsed time
        print(f"Execution time for '{func.__name__}': {execution_time:.4f} seconds")
        return result

    return wrapper


if __name__ == "__main__":
    theoretical_spectrum = pd.Series([100, 200, 300, 400, 500]).to_numpy()
    experimental_spectrum = pd.DataFrame({'mz': [100, 200, 300, 400, 500, 600, 700, 800, 900],
                                          'intensity': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5]}).to_numpy()
    t = 0.2


    @time_execution
    def score():
        score = simple_scoring_function(theoretical_spectrum, experimental_spectrum, t)
        print(score)


    score()
