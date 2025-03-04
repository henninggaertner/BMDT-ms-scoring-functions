import time
from functools import wraps
from scipy.stats import binom
import pandas as pd
import numpy as np
import math

pd.options.mode.chained_assignment = None  # default='warn'

# given: theoretical spectrum (only mz values), experimental spectrum (mz and intensity values)
# return: score of the theoretical spectrum against the experimental spectrum

# 1. divide the mz values of the theoretical spectrum into the 100Th windows and sort them by intensity
# 2. given q as maximum number of peaks in the window that can be matched, extract top q peaks for each window
# 3. match the peaks of the theoretical spectrum with the peaks of the experimental spectrum with tolerance t
# 4. calculate the score as the number of matched peaks divided by the total number of peaks in the theoretical spectrum

def scoring_function(theoretical_spectrum: np.ndarray, experimental_spectrum: np.ndarray, q: int, t: float) -> float:
    """
    Parameters:
    theoretical_spectrum : np.ndarray
        A 1D array containing the m/z values of the theoretical spectrum.
    experimental_spectrum : np.ndarray
        A 2D array with columns [m/z, intensity] representing the experimental spectrum.
    q : int
        The number of top intensity peaks to consider within each window of 100 Th.
    t : float
        Mass tolerance in parts per million (ppm).
    ----------
    Returns:
    float
        The negative logarithm (base 10) of the probability score.
        Higher values indicate a better match between the spectra.
        Returns -inf for extremely small probabilities.
    """
    
    # Ensure theoretical_spectrum is one-dimensional (m/z values only)
    if theoretical_spectrum.ndim != 1:
        raise ValueError("Theoretical spectrum must be a 1D array containing only m/z values.")

    # Sort the theoretical spectrum by m/z (no intensity available)
    theoretical_spectrum = np.sort(theoretical_spectrum)

    # Ensure experimental_spectrum is two-dimensional (m/z and intensity values)
    if experimental_spectrum.ndim != 2 or experimental_spectrum.shape[1] != 2:
        raise ValueError("Experimental spectrum must be a 2D array with columns [m/z, intensity].")

    # Sort the experimental spectrum by intensity (descending)
    experimental_spectrum = experimental_spectrum[np.argsort(experimental_spectrum[:, 1])[::-1]]

    # Divide the m/z values of the experimental spectrum into windows of 100 Th
    windows = np.floor(experimental_spectrum[:, 0] / 100)

    # Extract the top q peaks for each window
    unique_windows = np.unique(windows)
    filtered_experimental = []
    for window in unique_windows:
        peaks_in_window = experimental_spectrum[windows == window]
        filtered_experimental.append(peaks_in_window[:q])
    experimental_spectrum = np.vstack(filtered_experimental)

    # Determine n
    n = len(theoretical_spectrum)

    # Match peaks with tolerance t
    matched_peaks = []
    for peak in theoretical_spectrum:  # Iterate over m/z values in the theoretical spectrum
        allowed_deviation = peak * t / 1e6  # Convert tolerance from ppm to Da
        diff = np.abs(experimental_spectrum[:, 0] - peak)
        within_tolerance = diff <= allowed_deviation
        if np.any(within_tolerance):
            # Find the closest peak
            closest_index = np.argmin(diff[within_tolerance])
            closest_peak_index = np.where(within_tolerance)[0][closest_index]
            matched_peaks.append(experimental_spectrum[closest_peak_index])
            # Remove the matched peak from the experimental spectrum
            experimental_spectrum = np.delete(experimental_spectrum, closest_peak_index, axis=0)

    k = len(matched_peaks)
    # k=0 -> 1.0
    # k=1
    # k=n -> 0.00001 ( to n-1)


    if k == 0:
        probability = 1
    else:
        p = q / 100
        probability = 1- binom.cdf(k-1, n, p)
        if k == n:
            print(f"n: {n}, k: {k}, q: {q}, probability: {probability}")
    try:
        return -10 * math.log10(probability)
    except ValueError:
        return -np.inf



def cached_comb(n, k):
    # Efficiently compute binomial coefficients
    return math.comb(n, k)


def optimize_q_wrapper(theoretical_spectrum: np.ndarray, experimental_spectrum: np.ndarray, t: float) -> int:
    max_score = -np.inf
    for i in range(2, 101):
        score = scoring_function(theoretical_spectrum, experimental_spectrum, i, t)
        if score > max_score:
            max_score = score
        else:
            break
    return max_score


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
    theoretical_spectrum = pd.Series([100, 120, 130, 140, 200, 300, 400, 500]).to_numpy()
    experimental_spectrum = pd.DataFrame({'mz': [100, 120, 130, 140, 200, 300, 400, 500, 600, 700, 800, 900],
                                          'intensity': [0.1, 0.1, 0.1, 0.1, 0.1,
                                                                                                            0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5]}).to_numpy()
    t = 2

    @time_execution
    def score():
        for i in range(1000):
            score = optimize_q_wrapper(theoretical_spectrum, experimental_spectrum, t)

    score()