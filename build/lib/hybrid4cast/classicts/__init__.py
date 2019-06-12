import numpy as np
from numbers import Number
from statsmodels.tsa.stattools import pacf


def autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: time series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))

    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


def smooth_autocorr(series, lag, partial=False):
    """
    Autocorrelation for single data series but smoothed over 3 days
    :param series: time series
    :param lag: lag, days
    :return:
    """
    if not isinstance(lag, Number):
        return 0

    lag = int(lag)
    if lag > 0.8 * len(series):
        return 0

    if partial:
        pacf_coef = pacf(series, lag + 1, method='ols')
        c = pacf_coef[lag]
        c_m1 = pacf_coef[lag - 1]
        c_p1 = pacf_coef[lag + 1]
    else:
        c = autocorr(series, lag)
        c_m1 = autocorr(series, lag - 1)
        c_p1 = autocorr(series, lag + 1)

    return 0.5 * c + 0.25 * c_m1 + 0.25 * c_p1
