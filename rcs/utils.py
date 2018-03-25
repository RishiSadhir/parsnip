import pandas as pd
import numpy as np
import math


raise_power_v = np.vectorize(math.pow, otypes=[np.float64])


def logistic(arg):
    return np.exp(arg) / (1 + np.exp(arg))


logistic_v = np.vectorize(logistic)


def identity(arg):
    return arg


def odds(arr):
    num_ones = np.sum(arr)
    numerator = num_ones / arr.size
    denominator = (arr.size - num_ones) / arr.size
    if denominator == 0:
        return np.Inf
    return numerator / denominator


def log_odds(arr):
    arr_odds = odds(arr)
    if arr_odds == 0:
        return -np.Inf
    return math.log(arr_odds)


def init_list(size):
    return [None] * size


def infer_variable_type(series):
    if series.dtype == "object":
        return "multinomial"
    elif series.value_counts().size == 2:
        return "binary"
    return "continuous"


def bin_array(arr, bins=100):
        min = np.percentile(arr, .1)
        max = np.percentile(arr, 99.9)
        step = (max - min)/bins
        return pd.Series(np.arange(min, max, step))


def get_quantiles(num_quantiles):
        step = (1.0 / (num_quantiles+1)) * 100
        return np.arange(start=step, stop=100, step=step)[:num_quantiles]
