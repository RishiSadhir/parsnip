import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotnine as gg
import math


def cubic_spline(x, y, k):
    rcs = Rcs()
    rcs.knots = getKnots(x, k)
    rcs.design_matrix = cubic_design_matrix(x, rcs.knots)
    rcs.outcome = infer_variable_type(y)
    rcs.model = fit_linear_function(rcs.design_matrix, y, rcs.outcome)
    rcs.coefficients = rcs.model.params
    rcs.evaluator = cs_evaluator(rcs.coefficients, rcs.knots)
    rcs.plot = generate_plot(x, y, rcs)
    return rcs


def linear_spline(x, y, k):
    rcs = Rcs()
    rcs.knots = getKnots(x, k)
    rcs.design_matrix = linear_design_matrix(x, rcs.knots)
    rcs.outcome = infer_variable_type(y)
    rcs.model = fit_linear_function(rcs.design_matrix, y, rcs.outcome)
    rcs.coefficients = rcs.model.params
    rcs.evaluator = ls_evaluator(rcs.coefficients, rcs.knots)
    rcs.plot = generate_plot(x, y, rcs)
    return rcs


class Rcs(object):
    def __init__(self):
        self.knots = []
        self.coefficients = []
        self.model = None
        self.plot = None
        self.design_matrix = None
        self.evaluator = None
        self.outcome = None

    def plot(self, xlable=None, ylable=None, title=None):
        p = self._plot
        if xlable is not None:
            p += gg.xlab(xlable)
        if ylable is not None:
            p += gg.ylab(ylable)
        if title is not None:
            p += gg.ggtitle(title)
        return p

    def evaluate(self, x):
        xb = self.evaluator(x)
        if self.outcome == "binary":
            return logistic(xb)
        return xb

    def fit_statistics(self):
        print(self.model.summary())

    def vectorize_evaluator(self):
        return np.vectorize(self.evaluator)


def linear_design_matrix(x, knots):
    df = pd.DataFrame(x)
    for knot in knots:
        df["k_" + str(knot)] = segmentize(x, knot)
    return df


def cubic_design_matrix(x, knots):
    df = pd.DataFrame({"x1": x})
    df["x2"] = raise_power_v(x, 2)
    df["x3"] = raise_power_v(x, 3)
    for knot in knots:
        df["k_" + str(knot)] = raise_power_v(segmentize(x, knot), 3)
    return df


def raise_power(elt, exponent):
    return math.pow(elt, exponent)


raise_power_v = np.vectorize(raise_power)


def ls_evaluator(coefficients, knots):
    def evaluator(x):
        result = coefficients[0] + (coefficients[1] * x)
        for idx, knot in enumerate(knots):
            result += coefficients[idx + 2] * shift_by_knot(x, knot)
        return result
    return evaluator


def cs_evaluator(coefficients, knots):
    def evaluator(x):
        result = coefficients[0] + \
                 (coefficients[1] * x) + \
                 (coefficients[2] * (x ** 2)) + \
                 (coefficients[3] * (x ** 3))
        for idx, knot in enumerate(knots):
            result += coefficients[idx + 4] * \
                      raise_power_v(shift_by_knot(x, knot), 3)
        return result
    return evaluator


def logistic(xb):
    return np.exp(xb) / (1 + np.exp(xb))


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


def infer_x(arr):
    min = np.percentile(arr, .1)
    max = np.percentile(arr, 99.9)
    # Assume for now a fixed number of bins = 100
    step = (max - min)/100.0
    return pd.Series(np.arange(min, max, step))


def fit_linear_function(x, y, outcome_type):
    exog = sm.add_constant(x.as_matrix())
    endog = np.array(y)
    if outcome_type == "continous":
        return sm.OLS(endog, exog).fit()
    else:
        return sm.GLM(endog, exog, family=sm.families.Binomial()).fit()


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
