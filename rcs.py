# -*- coding: utf-8 -*-
"""Spline methods and objects

This module contains methods and classes that implement
linear, cubic, and restricted cubic splines. Access this
functionality through numpy arrays or pandas series / dataframes.


API:
    cv_spline - Cross validate knots of a spline
    spline - Train a prespecified spline
    splinify_df - Apply basis transform to a pandas dataframe

Spline object API
    knots
    model
    plot
    evaluate
    fit statistics

Todo:
    * Implement a log odds plot
    * Cross validation
    * Prespecified knots
    * Confidence intervals

    * PyTest coverage
    * Documentation
    * Restricted Cubic Spline implementation
    * Replace statsmodels with something better

    * Tensors
"""

# Data Structures
import pandas as pd
import numpy as np
import collections

# Regression
import statsmodels.api as sm

# Plotting
import plotnine as gg

# Functional programming
from functools import partial

# Local Imports
import utils


def cv_spline(x, y, k, type="rcs", num_folds=5):
    """ Cross validation to optimize knot placement
    """
    raise NotImplemented


def spline(x, y, k=5, type="rcs"):
    """ Train a 2 dimensional spline
    """
    if type == "rcs":
        return _restricted_cubic_spline(x, y, k)
    elif type == "cs":
        return _cubic_spline(x, y, k)
    else:
        return _linear_spline(x, y, k)


def splinify_df(df, cols, k, type="rcs"):
    """ Apply basis transform to vector
    """
    if type == "rcs":
        pass
    elif type == "cs":
        pass
    else:
        pass


def _restricted_cubic_spline(x, y, k):
    """ Train a restricted cubic spline
    """
    if k < 3:
        raise ValueError("Restricted cubic splines require at least 3 knots")
    return RestrictedCubicSpline(x, y, k, "rcs", True)


def _cubic_spline(x, y, k):
    """ Train a cubic spline
    """
    return CubicSpline(x, y, k, "cs", True)


def _linear_spline(x, y, k):
    """ Train a linear spline
    """
    return LinearSpline(x, y, k, "ls", True)


class Spline(object):
    """ A fit spline

    API
        knots
        model
        plot
        evaluate
        fit statistics
    """
    def __init__(self, x, y, k, spline_type, keep_data=True):
        self.knots = Spline._getKnots(x, k)
        self._design_matrix = self._create_design_matrix(x)
        self.outcome_type = utils.infer_variable_type(y)
        self.model = self._fit_linear_function(self._design_matrix, y)
        self.coefficients = self.model.params
        self._evaluate = self._evaluator()
        self._evaluate_v = np.vectorize(self._evaluator())
        if not keep_data:
            self._design_matrix = None
        self.plot = partial(self._generate_plot, x, y)

    def evaluate(self, x):
        is_iterable = isinstance(x, collections.Iterable)
        if is_iterable:
            yhat = self._evaluate_v(x)
        else:
            yhat = self._evaluate(x)
        if self.outcome_type == "binary":
            if is_iterable:
                return utils.logistic_v(yhat)
            return utils.logistic(yhat)
        return yhat

    def fit_statistics(self):
        print(self.model.summary())

    def _infer_x(arr):
        min = np.percentile(arr, .1)
        max = np.percentile(arr, 99.9)
        # Assume for now a fixed number of bins -> 100
        step = (max - min)/100.0
        return pd.Series(np.arange(min, max, step))

    def getQuantiles(k):
        step = (1.0 / (k+1)) * 100
        return np.arange(start=step, stop=100, step=step)[:k]

    def _evaluator(self):
        raise NotImplementedError(
            'subclasses must override create_design_matrix')

    def _create_design_matrix(self, x):
        raise NotImplementedError(
            'subclasses must override create_design_matrix')

    def _fit_linear_function(self, x, y):
        exog = sm.add_constant(x.as_matrix())
        endog = np.array(y)
        if self.outcome_type == "continuous":
            return sm.OLS(endog, exog).fit()
        else:
            return sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    def _segmentize(x, knot):
        return x.apply(lambda x: Spline._shift_by_knot(x, knot))

    def _shift_by_knot(elt, knot):
        elt = elt - knot
        if elt < 0:
            return 0
        return elt

    def _getKnots(x, k):
        quantiles = Spline.getQuantiles(k)
        knots = np.percentile(x, quantiles)
        return knots

    def _tau(self):
        k = len(self.knots)
        tau = self.knots[k-1] - self.knots[0]
        return tau * tau

    def _generate_plot(self, x, y, xlabel=None, ylabel=None, title=None):
        df = pd.DataFrame({
            "x": x,
            "y": y,
            "fit": self.evaluate(x)
        })

        p = gg.ggplot(df, gg.aes("x", "y"))

        # Add points to the continuous plot
        if self.outcome_type == "continuous":
            p += gg.geom_point(color="steelblue", alpha=1/4)

        # When the outcome is binary, use log odds
        #
        # There appears to be an ongoing bug in plotnine that is
        # Making the below not work
        # else:
        #     p += gg.stat_summary_bin(geom="point", fun_y=np.mean,
        #                              color="steelblue")

        p += gg.geom_rug(sides='b')
        plot_data = pd.DataFrame({
            "x_axis": Spline._infer_x(x),
            "y_axis": self.evaluate(Spline._infer_x(x))
        })
        p += gg.geom_line(data=plot_data, mapping=gg.aes("x_axis", "y_axis"),
                          size=1, color="black")
        for knot in self.knots:
            p += gg.geom_point(gg.aes(x=knot, y=self.evaluate(knot)),
                               shape="x", size=4, color="darkblue")

        if xlabel is not None:
            p += gg.xlab(xlabel)
        if ylabel is not None:
            p += gg.ylab(ylabel)
        if title is not None:
            p += gg.ggtitle(title)
        return p


class RestrictedCubicSpline(Spline):
    def _create_design_matrix(self, x):
        df = pd.DataFrame({"x1": x})
        for idx, knot in enumerate(self.knots[:-2]):
            t1 = utils.raise_power_v(Spline._segmentize(x, knot), 3)
            t2 = utils.raise_power_v(Spline._segmentize(x, self.knots[-2]), 3) *\
                 (self.knots[-1] - knot) / (self.knots[-1] - self.knots[-2])
            t3 = utils.raise_power_v(Spline._segmentize(x, self.knots[-1]), 3) *\
                 (self.knots[-2] - knot) / (self.knots[-1] - self.knots[-2])
            vec = t1 - t2 + t3
            df["X_" + idx+2] = vec / self._tau()

    def _evaluator(self):
        # def evaluator_fn(x):
        #     result = self.coefficients[0] +\
        #              (self.coefficients[1] * x)
        #     for
        pass


class CubicSpline(Spline):
    def _create_design_matrix(self, x):
        df = pd.DataFrame({"x1": x})
        df["x2"] = utils.raise_power_v(x, 2)
        df["x3"] = utils.raise_power_v(x, 3)
        for knot in self.knots:
            df["k_" + str(knot)] = utils.raise_power_v(
                Spline._segmentize(x, knot), 3)
        return df

    def _evaluator(self):
        def evaluator_fn(x):
            result = self.coefficients[0] +\
                     (self.coefficients[1] * x) + \
                     (self.coefficients[2] * (x ** 2)) + \
                     (self.coefficients[3] * (x ** 3))
            for idx, knot in enumerate(self.knots):
                result += self.coefficients[idx + 4] * \
                          utils.raise_power_v(Spline._shift_by_knot(x, knot), 3)
            return result
        return evaluator_fn


class LinearSpline(Spline):
    def _create_design_matrix(self, x):
        df = pd.DataFrame(x)
        for knot in self.knots:
            df["k_" + str(knot)] = Spline._segmentize(x, knot)
        return df

    def _evaluator(self):
        def evaluator_fn(x):
            result = self.coefficients[0] + (self.coefficients[1] * x)
            for idx, knot in enumerate(self.knots):
                result += self.coefficients[idx + 2] * \
                          Spline._shift_by_knot(x, knot)
            return result
        return evaluator_fn
