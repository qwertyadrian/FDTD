# -*- coding: utf-8 -*-
"""
Модуль со вспомогательными классами и функциями, не связанные напрямую с
методом FDTD
"""

from typing import List, Tuple

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from numpy.fft import fft, fftshift


class Probe:
    """
    Класс для хранения временного сигнала в датчике.
    """
    def __init__(self, position: int, maxTime: int):
        """
        position - положение датчика (номер ячейки).
        maxTime - максимально количество временных
            шагов для хранения в датчике.
        """
        self.position = position

        # Временные сигналы для полей E и H
        self.E = np.zeros(maxTime)
        self.H = np.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E: npt.NDArray, H: npt.NDArray):
        """
        Добавить данные по полям E и H в датчик.
        """
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1


class AnimateFieldDisplay:
    """
    Класс для отображения анимации распространения ЭМ волны в пространстве
    """

    def __init__(self,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str):
        """
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y
        """
        self.maxXSize = maxXSize
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, отсчет'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'
        self._fig = None
        self._ax = None

    def activate(self):
        """
        Инициализировать окно с анимацией
        """
        self._xList = np.arange(self.maxXSize)

        # Создание окна для графика
        self._fig, self._ax = plt.subplots()
        self._fig.set_dpi(100)

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, np.zeros(self.maxXSize))

    def drawProbes(self, probesPos: List[int]):
        """
        Нарисовать датчики.

        probesPos - список координат датчиков для регистрации временных
            сигналов (в отсчетах).
        """
        # Отобразить положение датчиков
        self._ax.plot(probesPos, [0] * len(probesPos), self._probeStyle)

    def drawSources(self, sourcesPos: List[int]):
        """
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        """
        # Отобразить положение источников
        self._ax.plot(sourcesPos, [0] * len(sourcesPos), self._sourceStyle)

    def drawBoundaries(self, *positions: int):
        """
        Нарисовать границы в области моделирования.

        positions - координаты X границы (в отсчетах).
        """
        for pos in positions:
            self._ax.plot([pos, pos],
                          [self.minYSize, self.maxYSize],
                          '--k')

    def updateData(self, data: npt.NDArray) -> Tuple[matplotlib.lines.Line2D]:
        """
        Обновить данные с распределением поля в пространстве
        """
        self._line.set_ydata(data)
        return self._line,

    def start_animation(self, Ez: List[npt.NDArray]) -> FuncAnimation:
        """
        Создание анимации
        """
        return FuncAnimation(
            self._fig,
            self.updateData,
            frames=Ez,
            blit=True,
            interval=40,
            repeat=False,
        )


def showProbeSignals(
        probes: List[Probe], minYSize: float, maxYSize: float, maxXsize=None,
        filename="probe_signals.png"
):
    """
    Показать графики сигналов, зарегистрированных в датчиках.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    """
    # Создание окна с графиков
    fig, ax = plt.subplots()

    # Настройка внешнего вида графиков
    ax.set_xlim(0, maxXsize if maxXsize else probes[0].E.size)
    ax.set_ylim(minYSize, maxYSize)
    ax.set_xlabel('q, отсчет')
    ax.set_ylabel('Ez, В/м')
    ax.grid()

    # Вывод сигналов в окно
    for probe in probes:
        ax.plot(probe.E)

    # Создание и отображение легенды на графике
    legend = ['Probe x = {}'.format(probe.position) for probe in probes]
    ax.legend(legend)

    # Сохранить график
    fig.savefig(filename)
    plt.show()


def show_signal_spectrum(
        probes: List[Probe], dt,
        xmin=0, xmax=1e9, filename="signal_spectrum.png"
):
    spectrums = [fftshift(np.abs(fft(probe.E))) for probe in probes]
    size = probes[0].E.size
    df = 1.0 / (size * dt)
    freq = np.arange(-size / 2 * df, size / 2 * df, df)
    fig, ax = plt.subplots()
    for spectrum in spectrums:
        ax.plot(freq, spectrum / np.max(spectrum))
    ax.grid()
    ax.set_xlabel('Частота, Гц')
    ax.set_ylabel('|S| / |Smax|')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    fig.savefig(filename)
    plt.show()


def show_reflection_coeff(
        fail_signal: npt.NDArray, reflected_signal: npt.NDArray, dt: float,
        xmin=0.2e9, xmax=2e9,
        filename: str = "reflection_coeff.png"
):
    fig, ax = plt.subplots()
    spectrum1 = fftshift(np.abs(fft(fail_signal)))
    spectrum2 = fftshift(np.abs(fft(reflected_signal)))
    size = fail_signal.size
    df = 1.0 / (size * dt)
    freq = np.arange(-size / 2 * df, size / 2 * df, df)
    ax.plot(freq, spectrum2 / spectrum1)
    ax.grid()
    ax.set_xlabel('Частота, Гц')
    ax.set_ylabel(r"$|\Gamma|$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 0.7)
    fig.savefig(filename)
    plt.show()
