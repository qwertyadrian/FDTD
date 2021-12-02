from abc import ABCMeta, abstractmethod

import numpy as np


class SourceBase(metaclass=ABCMeta):
    @abstractmethod
    def getE(self, m, q):
        pass


class GaussianDiffPlaneWave(SourceBase):
    """
    Класс с уравнением плоской волны
    для дифференцированного гауссова сигнала в дискретном виде
    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды,
    в которой расположен источник.
    mu - относительная магнитная проницаемость среды,
    в которой расположен источник.
    """

    def __init__(self, d, w, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d
        self.w = w
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        """
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        """
        tmp = ((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w
        return -2 * tmp * np.exp(-(tmp ** 2))


class RickerPlaneWave(SourceBase):
    def __init__(self, Np, Md, Sc=1.0, eps=1.0, mu=1.0):
        self.Np = Np
        self.Md = Md
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        tmp = (self.Sc * q / self.Np - self.Md)**2
        return (1 - 2 * np.pi**2 * tmp) * np.exp(-np.pi**2 * tmp)
