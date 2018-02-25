import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotnine as gg
import utils


def cv_spline(x, y, k, type="rcs", num_folds=5):
    raise NotImplemented


def spline(x, y, k, type="rcs"):
    if type == "rcs":
        return restricted_cubic_spline(x, y, k)
    elif type == "cs":
        return cubic_spline(x, y, k)
    else:
        return linear_spline(x, y, k)


def restricted_cubic_spline(x, y, k):
    return RestrictedCubicSpline(x, y, k, "cs", True)


def cubic_spline(x, y, k):
    return CubicSpline(x, y, k, "cs", True)


def linear_spline(x, y, k):
    return LinearSpline(x, y, k, "ls", True)


class Spline(object):
    def __init__(self, x, y, k, spline_type, keep_data=True):
        self.knots = Spline._getKnots(x, k)
        design_matrix = self._create_design_matrix(x, self.knots)
        self.outcome_type = utils.infer_variable_type(y)
        self.model = self.fit_linear_function(design_matrix, y)
        self.coefficients = self.model.params
        self.evaluator = self.evaluator()
        self.plot = self.generate_plot(x, y)
        del design_matrix

    def evaluator(self):
        raise NotImplementedError(
            'subclasses must override create_design_matrix')

    def evaluate(self, x):
        xb = self.evaluator(x)
        if self.outcome_type == "binary":
            return utils.logistic(xb)
        return xb

    def fit_statistics(self):
        print(self.model.summary())

    def vectorize_evaluator(self):
        return np.vectorize(self.evaluator)

    def _create_design_matrix(self, x):
        raise NotImplementedError(
            'subclasses must override create_design_matrix')

    def fit_model(self, y):
        raise NotImplementedError(
            'subclasses must override create_design_matrix')

    def fit_linear_function(self, x, y):
        exog = sm.add_constant(x.as_matrix())
        endog = np.array(y)
        if self.outcome_type == "continous":
            return sm.OLS(endog, exog).fit()
        else:
            return sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    def segmentize(x, knot):
        return x.apply(lambda x: Spline.shift_by_knot(x, knot))

    def shift_by_knot(elt, knot):
        elt = elt - knot
        if elt < 0:
            return 0
        return elt

    def _getKnots(x, k):
        quantiles = Spline.getQuantiles(k)
        knots = np.percentile(x, quantiles)
        return knots

    def getQuantiles(k):
        step = (1.0 / (k+1)) * 100
        return np.arange(start=step, stop=100, step=step)[:k]

    def _infer_x(arr):
        min = np.percentile(arr, .1)
        max = np.percentile(arr, 99.9)
        # Assume for now a fixed number of bins = 100
        step = (max - min)/100.0
        return pd.Series(np.arange(min, max, step))

    def generate_plot(self, x, y, xlabel=None, ylabel=None, title=None):
        df = pd.DataFrame({
            "x": x,
            "y": y,
            "fit": x.apply(self.evaluate)
        })
        p = gg.ggplot(df, gg.aes("x", "y"))
        # Add points
        if self.outcome_type == "continous":
            p += gg.geom_point(color="steelblue", alpha=1/4)
            # When the outcome is binary, use log odds
        else:
            p += gg.stat_summary(geom="point", fun_y=utils.log_odds,
                                 color="steelblue")
        plot_data = pd.DataFrame({
            "x_axis": Spline._infer_x(x),
            "y_axis": Spline._infer_x(x).apply(self.evaluate)
        })
        p += gg.geom_line(data=plot_data, mapping=gg.aes("x_axis", "y_axis"),
                          size=1, color="black")
        p += gg.geom_rug(sides='b')
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
    raise NotImplementedError


class CubicSpline(Spline):
    def _create_design_matrix(self, x):
        df = pd.DataFrame({"x1": x})
        df["x2"] = utils.raise_power_v(x, 2)
        df["x3"] = utils.raise_power_v(x, 3)
        for knot in self.knots:
            df["k_" + str(knot)] = utils.raise_power_v(
                Spline.segmentize(x, knot), 3)
        return df

    def evaluator(self):
        def evaluation_fn(x):
            result = self.coefficients[0] +\
                     (self.coefficients[1] * x) + \
                     (self.coefficients[2] * (x ** 2)) + \
                     (self.coefficients[3] * (x ** 3))
            for idx, knot in enumerate(self.knots):
                result += self.coefficients[idx + 4] * \
                          utils.raise_power_v(Spline.shift_by_knot(x, knot), 3)
            return result
        return evaluation_fn


class LinearSpline(Spline):
    def _create_design_matrix(self, x):
        df = pd.DataFrame(x)
        for knot in self.knots:
            df["k_" + str(knot)] = Spline.segmentize(x, knot)
        return df

    def ls_evaluator(self):
        def evaluator_fn(x):
            result = self.coefficients[0] + (self.coefficients[1] * x)
            for idx, knot in enumerate(self.knots):
                result += self.coefficients[idx + 2] * \
                          Spline.shift_by_knot(x, knot)
            return result
        return evaluator_fn

