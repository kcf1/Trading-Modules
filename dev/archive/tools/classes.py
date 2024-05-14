import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt


class StrategyEvaluator:
    """
    Evaluate daily PnL
    Trading freqency < daily will be summed to daily PnL
    """

    def __init__(self, pnl: pd.Series):
        self.pnl = pnl.resample("d").sum()
        self.ann_scaler = 252
        self.test_days = len(self.pnl)

    def plot(self) -> None:
        """
        Plot equity curve
        """
        ax = plt.subplot()
        ax.set_title("Equity curve")
        self.pnl.cumsum().plot(ax=ax)
        plt.show()

    def get_stats(self) -> pd.Series:
        """
        Distribution stats
        Risk metrics
        Ratios
        """
        dists = self.get_dists()
        risks = self.get_risks()
        ratios = self.get_ratios()
        stats = pd.concat([dists, risks, ratios])
        return stats

    def get_dists(self) -> pd.Series:
        """
        Distribution statistics (ann.)
        """
        ann = self.ann_scaler

        self.mean = self.pnl.mean() * ann
        self.vol = self.pnl.std() * sqrt(ann)
        self.median = self.pnl.median() * ann
        self.lowqt = self.pnl.quantile(q=0.25) * ann
        self.upqt = self.pnl.quantile(q=0.75) * ann
        self.skew = self.pnl.skew()
        self.set_skew_monthly()
        self.kurt = self.pnl.kurt()
        self.set_tail_ratios()
        dists = pd.Series(
            {
                "mean": self.mean,
                "vol": self.vol,
                "median": self.median,
                "lowqt": self.lowqt,
                "upqt": self.upqt,
                "skew": self.skew,
                "skew_monthly": self.skew_monthly,
                "lowtail": self.lowtail,
                "uptail": self.uptail,
                "kurt": self.kurt,
            }
        )
        return dists

    def get_risks(self) -> pd.Series:
        """
        Risk metrics
        """
        self.set_drawdown_metrics()
        self.set_downsiderisk()
        self.set_VaR_metrics()
        risks = pd.Series(
            {
                "avgdd": self.avgdd,
                "maxdd": self.maxdd,
                "mdd_duration": self.mdd_duration,
                "mdd_peak_idx": self.mdd_peak_idx,
                "mdd_trough_idx": self.mdd_trough_idx,
                "downsiderisk": self.downsiderisk,
                "Ulcer": self.Ulcer,
                "VaR95": self.VaR95,
                "VaR99": self.VaR99,
                "expsf95": self.expsf95,
                "expsf99": self.expsf99,
            }
        )
        return risks

    def get_ratios(self) -> pd.Series:
        """
        Ratios
        """
        self.profitfactor = self.pnl.clip(lower=0).sum() / -self.pnl.clip(upper=0).sum()
        self.Sharpe = self.mean / self.vol
        self.Sortino = self.mean / self.downsiderisk
        self.Calmar = self.mean / self.maxdd
        self.Ulcerperf = self.mean / self.Ulcer
        ratios = pd.Series(
            {
                "profitfactor": self.profitfactor,
                "Sharpe": self.Sharpe,
                "Sortino": self.Sortino,
                "Calmar": self.Calmar,
                "Ulcerperf": self.Ulcerperf,
            }
        )
        return ratios

    def set_drawdown_metrics(self) -> pd.Series:
        """
        Average drawdown, maximum drawdown, max drawdown duration
        """
        cum_pnl = self.pnl.cumsum()
        drawdown = cum_pnl - cum_pnl.expanding().max()
        drawdown = drawdown[drawdown < 0]
        self.avgdd = drawdown.mean()
        self.maxdd = drawdown.min()

        self.Ulcer = sqrt((drawdown**2).mean())

        self.mdd_trough_idx = cum_pnl[cum_pnl == self.maxdd].index[0]
        self.mdd_peak_idx = cum_pnl.loc[: self.mdd_trough_idx][cum_pnl == cum_pnl.max()]
        self.mdd_duration = len(cum_pnl.loc[self.mdd_peak_idx : self.mdd_trough_idx])

    def set_downsiderisk(self) -> None:
        """
        Downside deviation from 0
        """
        downside = self.pnl.clip(upper=0)
        self.downsiderisk = sqrt((downside**2).mean()) * self.ann_scaler

    def set_VaR_metrics(self) -> None:
        """
        Historical 95,99% VaR and expected shortfall
        """
        self.VaR95 = self.pnl.quantile(0.05)
        self.VaR99 = self.pnl.quantile(0.01)
        self.expsf95 = self.pnl[self.pnl <= self.VaR95].mean()
        self.expsf99 = self.pnl[self.pnl <= self.VaR99].mean()

    def set_skew_monthly(self) -> float:
        """
        Skewness of resampled monthly PnL
        """
        pnl_monthly = self.pnl.resample("m").sum()
        self.skew_monthly = pnl_monthly.skew()

    def set_tail_ratios(self) -> pd.Series:
        """
        Tail-ratios by Rob Carver
        """
        demean_pnl = self.pnl - self.pnl.mean()
        self.lowtail = (demean_pnl.quantile(0.01) / demean_pnl.quantile(0.30)) / 4.43
        self.uptail = (demean_pnl.quantile(0.99) / demean_pnl.quantile(0.70)) / 4.43
