import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotnine as gg
import toolz as fn
import scipy.stats as stats
import math


def linearSplines(x, y, k):
    rcs = Rcs()
    rcs.knots = getKnots(x, k)

    design_matrix = pd.DataFrame(x)
    for knot in rcs.knots:
        design_matrix["k_" + str(knot)] = segmentize(x, knot)
    # To be removed
    rcs.design_matrix = design_matrix
    rcs.outcome = infer_variable_type(y)
    rcs.model = fit_linear_function(design_matrix, y, rcs.outcome)
    rcs.coefficients = rcs.model.params
    rcs.plot = generate_plot(x, y, rcs)
    return rcs


def generate_plot(x, y, mod):
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "fit": x.apply(mod.evaluate)
    })
    p = gg.ggplot(df, gg.aes("x", "y"))
    # Add points
    if mod.outcome == "continous":
        p += gg.geom_point(color="steelblue", alpha=1/4)
    # When the outcome is binary, use log odds
    else:
        p += gg.stat_summary(geom="point", color="steelblue", fun_y=log_odds)
    plot_data = pd.DataFrame({
        "x_axis": infer_x(x),
        "y_axis": infer_x(x).apply(mod.evaluate)
    })
    p += gg.geom_line(data=plot_data, mapping=gg.aes("x_axis", "y_axis"),
                      size=1, color="black")
    p += gg.geom_rug(sides='b')
    for knot in mod.knots:
        p += gg.geom_point(gg.aes(x=knot, y=mod.evaluate(knot)),
                           shape="x", size=4, color="darkblue")
    return p


def odds(arr):
    num_ones = np.sum(arr)
    numerator = num_ones / arr.size
    denominator = (arr.size - num_ones) / arr.size
    if denominator == 0:
        return np.Inf
    return numerator / denominator


def log_odds(arr):
    def protected_log(elt):
        if elt == 0:
            return -np.Inf
        else:
            return math.log(elt)
    return protected_log(odds(arr))


def infer_x(arr):
    min = np.percentile(arr, .1)
    max = np.percentile(arr, 99.9)
    # Assume for now a fixed number of bins = 100
    step = (max - min)/100.0
    return pd.Series(np.arange(min, max, step))


class Rcs(object):
    def __init__(self):
        self.knots = []
        self.coefficients = []
        self.model = None
        self.plot = None
        self.design_matrix = None

    def evaluate(self, x):
        result = self.coefficients[0]
        result += x * self.coefficients[1]
        for idx, knot in enumerate(self.knots):
            result += shift_by_knot(x, knot) * self.coefficients[idx+2]
        return result

    def plot(self):
        return self.plot


def fit_linear_function(x, y, outcome_type):
    input = sm.add_constant(x.as_matrix())
    outcome = np.array(y)
    if outcome_type == "continous":
        return sm.OLS(outcome, input).fit()
    else:
        return sm.GLM(outcome, input, family=sm.families.Binomial()).fit()


def segmentize(x, knot):
    return x.apply(lambda x: shift_by_knot(x, knot))


def shift_by_knot(elt, knot):
    elt = elt - knot
    if elt < 0:
        return 0
    return elt


def getKnots(x, k):
    quantiles = getQuantiles(k)
    knots = np.percentile(x, quantiles)
    return knots


def getQuantiles(k):
    step = (1.0 / (k+1)) * 100
    return np.arange(start=step, stop=100, step=step)[:k]


def initList(size):
    return [None] * size


def infer_variable_type(series):
    if series.dtype == "object":
        return "multinomial"
    elif series.value_counts().size == 2:
        return "binary"
    return "continous"
