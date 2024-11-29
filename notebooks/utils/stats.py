import pandas
from scipy.stats import kstest

P_VALUE_THRESHOLD = 0.05


class KSTest:
    """
    This class performs the Kolmogorov-Smirnov test to check if a given
    feature is normally distributed.
    """

    def __init__(self, data: pandas.DataFrame):
        self.data = data
        self.p_values = {}
        self._run_test()

    def _run_test(self):
        for feature in self.data.columns:
            _, p_value = kstest(self.data[feature], "norm")
            self.p_values[feature] = p_value

    def get_p_values(self):
        return self.p_values

    def is_feature_normal(self, feature: str) -> bool:
        return self.p_values[feature] > P_VALUE_THRESHOLD


def detect_outliers(data: pandas.Series) -> dict:
    """
    This method detects outliers in a given data using
    the Inter Quartile Range (IQR) method.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return {
        "outliers": outliers,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "percentage": len(outliers) / len(data) * 100,
    }
