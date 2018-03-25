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

import math

# Data Structures
import pandas as pd
import numpy as np
import collections

# Regression
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score

# Plotting
import plotnine as gg

# Functional programming
from functools import partial

# Local Imports
from . import utils


# Stone and koo's recommended quantiles
QUANTILE_MAP = {
    3: [10, 50, 90],
    4: [5, 35, 65, 95],
    5: [5, 27.5, 50, 72.5, 95],
    6: [5, 23, 41, 59, 77, 95],
    7: [2.5, 18.33, 34.17, 50, 65.83, 81.67, 97.5]}


def cv_spline(x, y, knot_grid, spline_type="rcs", num_folds=5, metric=None):
    """ Cross validation to optimize knot placement
    """
    results = {}
    kf = KFold(n_splits=num_folds)
    for knot_candidate in knot_grid:
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fit = spline(x_train, y_train, knot_candidate, spline_type)
            yhat = fit.predict(x_test, y_test)
            if metric is None:
                if fit.outcome_type == "continuous":
                    metric = mean_squared_error
                else:
                    metric = roc_auc_score
            results[knot_candidate] = metric(y_test, yhat)
    return results


def spline(x, y, k=5, spline_type="rcs"):
    """ Train a 2 dimensional spline
    """
    if not isinstance(k, collections.Iterable):
        k = get_knots(x, k)
    else:
        k = np.unique(k)
    if len(x) < len(k):
        raise ValueError("Fewer knots than data points.")
    if spline_type == "rcs":
        return restricted_cubic_spline(x, y, k)
    elif spline_type == "cs":
        return cubic_spline(x, y, k)
    else:
        return linear_spline(x, y, k)


def splinify(x, k, spline_type="rcs"):
    """ Apply basis transform to vector
    """
    if not isinstance(k, collections.Iterable):
        k = get_knots(x, k)
    if spline_type == "rcs":
        return rcs_basis_transform(x, k)
    elif spline_type == "cs":
        return cs_basis_transform(x, k)
    else:
        return linear_basis_transform(x, k)


def restricted_cubic_spline(x, y, k):
    """ Train a restricted cubic spline
    """
    if len(k) < 3:
        raise ValueError("Restricted cubic splines require at least 3 knots")
    return RestrictedCubicSpline(x, y, k, "rcs", True)


def cubic_spline(x, y, k):
    """ Train a cubic spline
    """
    if len(k) < 3:
        raise ValueError("Restricted cubic splines require at least 3 knots")
    return CubicSpline(x, y, k, "cs", True)


def linear_spline(x, y, k):
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
        self.knots = k
        self.design_matrix = self._create_design_matrix(x)
        self.outcome_type = utils.infer_variable_type(y)
        self.model = fit(self.design_matrix, y, self.outcome_type)
        self.coefficients = self.model.params
        self._evaluate_v = np.vectorize(self._evaluate)
        if not keep_data:
            self._design_matrix = None
        self.plot = partial(self._generate_plot, x, y)

    def _evaluate(x):
        raise NotImplementedError(
            'subclasses must override _evaluate')

    def _create_design_matrix(x):
        raise NotImplementedError(
            'subclasses must override _create_design_matrix')

    def fit_statistics(self):
        print(self.model.summary())

    def predict(self, x):
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

    def _generate_plot(self, x, y, xlabel=None, ylabel=None, title=None):
        df = pd.DataFrame({
            "x": x,
            "y": y,
            "fit": self.predict(x)})

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
            "x_axis": utils.bin_array(x),
            "y_axis": self.predict(utils.bin_array(x))})
        p += gg.geom_line(data=plot_data, mapping=gg.aes("x_axis", "y_axis"),
                          size=1, color="black")
        for knot in self.knots:
            p += gg.geom_point(gg.aes(x=knot, y=self.predict(knot)),
                               shape="x", size=4, color="darkblue")

        if xlabel is not None:
            p += gg.xlab(xlabel)
        if ylabel is not None:
            p += gg.ylab(ylabel)
        if title is not None:
            p += gg.ggtitle(title)
        return p


class LinearSpline(Spline):
    def _create_design_matrix(self, x):
        return linear_basis_transform(x, self.knots)

    def _evaluate(self, x):
        return ls_evaluate(self.coefficients, self.knots, x)


class CubicSpline(Spline):
    def _create_design_matrix(self, x):
        return cs_basis_transform(x, self.knots)

    def _evaluate(self, x):
        return cs_evaluate(self.coefficients, self.knots, x)


class RestrictedCubicSpline(Spline):
    def __init__(self, x, y, k, spline_type, keep_data=True):
        super(RestrictedCubicSpline, self).__init__(x, y, k, spline_type, keep_data)
        tau = math.pow((self.knots[-1] - self.knots[0]), 2)
        coefficients = self.model.params
        coefficients = list(coefficients[0:2]) + [c/tau for c in coefficients[2:]]
        bk = beta_k(coefficients, self.knots)
        bk1 = beta_k_1(coefficients, self.knots)
        self.coefficients = list(coefficients) + [bk, bk1]

    def _create_design_matrix(self, x):
        return rcs_basis_transform(x, self.knots)

    def _evaluate(self, x):
        coefficients = self.coefficients
        knots = self.knots
        return rcs_evaluate(coefficients, knots, x)


def beta_k(coefficients, knots):
    estimate = 0
    for idx, beta in enumerate(coefficients[2:]):
        estimate = estimate + beta * (knots[idx] - knots[-1])
    return estimate / (knots[-1] - knots[-2])


def beta_k_1(coefficients, knots):
    estimate = 0
    for idx, beta in enumerate(coefficients[2:]):
        estimate = estimate + beta * (knots[idx] - knots[-2])
    return estimate / (knots[-2] - knots[-1])


def ls_evaluate(coefficients, knots, x):
    prediction = coefficients[0] + (coefficients[1] * x)
    for idx, knot in enumerate(knots):
        prediction += coefficients[idx + 2] * shift_by_knot(x, knot)
    return prediction


def cs_evaluate(coefficients, knots, x):
    prediction = coefficients[0] + (coefficients[1] * x) +\
             (coefficients[2] * (x ** 2)) +\
             (coefficients[3] * (x ** 3))
    for idx, knot in enumerate(knots):
        prediction += coefficients[idx + 4] * truncated_power_basis(x, knot)
    return prediction


def rcs_evaluate(coefficients, knots, x):
    prediction = coefficients[0] + (coefficients[1] * x)
    for idx, coefficient in enumerate(coefficients[2:]):
        prediction += coefficient * truncated_power_basis(x, knots[idx])
    return prediction


def linear_basis_transform(x, knots):
    df = pd.DataFrame(x)
    for knot in knots:
        df["k_" + str(knot)] = truncated_power_basis_v(x, knot, 1, 1)
    return df


def cs_basis_transform(x, knots):
    df = pd.DataFrame({"x1": x})
    df["x2"] = utils.raise_power_v(x, 2)
    df["x3"] = utils.raise_power_v(x, 3)
    for knot in knots:
        df["k_" + str(knot)] = truncated_power_basis_v(x, knot)
    return df


def rcs_basis_transform(x, knots):
    """ Testing function

    x : Series
    y : Series

    Uses the truncated power basis to generate a design matrix
    that can be used with a GLM to create a restricted cubic spline
    """
    norm = utils.raise_power_v(knots[-1] - knots[0], 2.0/3.0)

    df = pd.DataFrame({"X_1": x})
    for idx, knot in enumerate(knots[:-2]):
        # Term one
        t1 = truncated_power_basis_v(x, knot, norm, 3)
        # Term two
        t2 = truncated_power_basis_v(x, knots[-2], norm, 3)
        t2 = t2 * (knots[-1] - knot) / (knots[-1] - knots[-2])
        # Term three
        t3 = truncated_power_basis_v(x, knots[-1], norm, 3)
        t3 = t3 * (knots[-2] - knot) / (knots[-1] - knots[-2])

        df["X_" + str(idx+2)] = (t1 - t2 + t3)
    return df


def get_knots(x, k):
    if k in QUANTILE_MAP.keys():
        quantiles = QUANTILE_MAP.get(k)
        knots = np.percentile(x, quantiles)
        # Do a better job with outter knots when there
        # are fewer than 100 data points to go off of
        if len(x) < 100:
            knots[0] = x[4]
            knots[-1] = x[len(x)-5]
    else:
        quantiles = utils.get_quantiles(k)
        knots = np.percentile(x, quantiles)
    return knots


def fit(x, y, outcome_type):
        exog = sm.add_constant(x.as_matrix())
        endog = np.array(y)
        if outcome_type == "continuous":
            return sm.OLS(endog, exog).fit()
        else:
            return sm.GLM(endog, exog, family=sm.families.Binomial()).fit()


def truncated_power_basis_v(vec, knot, norm=1, power=3):
    return utils.raise_power_v(shift_by_knot_v(vec, knot) / norm, power)


def truncated_power_basis(x, knot, norm=1, power=3):
    return math.pow(shift_by_knot(x, knot) / norm, power)


def shift_by_knot(elt, knot):
    elt = elt - knot
    if elt < 0:
        return 0
    return elt


shift_by_knot_v = np.vectorize(shift_by_knot, otypes=[np.float64])
