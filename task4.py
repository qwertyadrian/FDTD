# -*- coding: utf-8 -*-
import pathlib

import numpy as np
from scipy.constants import speed_of_light

import tools
from sources import RickerPlaneWave

if __name__ == '__main__':
    # Размер области моделирования вдоль оси X (м)
    X = 1
    # Относительная диэлектрическая проницаемость области моделирования
    EPS0 = 1
    EPS1 = 5.5
    EPS2 = 2.3
    EPS3 = 1
    # Толщины слоев диэлектриков
    d1 = 0.3
    d2 = 0.2
    # Время расчета в отсчетах
    maxTime = 4000
    # Размер области моделирования в отсчетах
    maxSize = 250
    # Число Куранта
    Sc = 1.0
    # Размер пространственного шага
    delta_x = X / maxSize
    # Размер временного шага
    delta_t = Sc * delta_x / speed_of_light

    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Положение источника
    sourcePos = 50

    # Датчики для регистрации поля
    probesPos = [25]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Положение начала диэлектрика
    layer_x = 100
    layer_x1 = layer_x + int(d1 / delta_x)
    layer_x2 = layer_x1 + int(d2 / delta_x)

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[:layer_x] = EPS0
    eps[layer_x:layer_x1] = EPS1
    eps[layer_x1:layer_x2] = EPS2
    eps[layer_x2:] = EPS3
    # Магнитная проницаемость
    mu = np.ones(maxSize)

    # Потери в среде.
    loss = np.zeros(maxSize)
    loss[-25:] = 0.05
    loss[:25] = 0.05

    # Коэффициенты для расчета поля E
    ceze = (1 - loss) / (1 + loss)
    cezh = W0 / (eps * (1 + loss))

    # Коэффициенты для расчета поля H
    chyh = (1 - loss) / (1 + loss)
    chye = 1 / (W0 * (1 + loss))

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)
    # Массив, содержащий падающий сигнал
    Ez0 = np.zeros(maxTime)

    source = RickerPlaneWave(30.0, 1.0)

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = chyh[:-1] * Hy + chye[:-1] * (Ez[1:] - Ez[:-1])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = ceze[1:-1] * Ez[1:-1] + cezh[1:-1] * (Hy[1:] - Hy[:-1])

        Ez0[q] = (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                  source.getE(-0.5, q + 0.5))
        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += Ez0[q]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

    Ez0_probe = tools.Probe(sourcePos, maxTime)
    Ez0_probe.E = Ez0
    probes.append(Ez0_probe)
    # Путь к папке с результатами
    dir_ = pathlib.Path("results")
    # Создаем папку, если она не существует
    dir_.mkdir(exist_ok=True)

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(
        probes, -1.1, 1.1, maxXsize=1000,
        filename="results/task4_probeSignals.png"
    )

    tools.show_signal_spectrum(
        probes, delta_t, xmax=10e9, filename="results/task4_signalSpectrum.png"
    )

    tools.show_reflection_coeff(
        Ez0, probes[0].E, delta_t, filename="results/task4_reflectionCoeff.png"
    )
