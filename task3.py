# -*- coding: utf-8 -*-
import pathlib

import numpy as np
from scipy.constants import speed_of_light

import tools
from sources import GaussianDiffPlaneWave

if __name__ == '__main__':
    # Размер области моделирования вдоль оси X
    X = 3.5
    # Относительная диэлектрическая проницаемость области моделирования
    EPS = 8.0
    # Время расчета в отсчетах
    maxTime = 1000
    # Размер области моделирования в отсчетах
    maxSize = 200
    # Скорость распространения волны в диэлектрике
    speed = speed_of_light / np.sqrt(EPS)
    # Размер пространственного шага
    delta_x = X / maxSize
    # Размер временного шага
    delta_t = delta_x / speed

    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Число Куранта
    Sc = (speed * delta_t) / delta_x

    # Положение источника
    sourcePos = 100

    # Датчики для регистрации поля
    probesPos = [150]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Положение начала диэлектрика
    layer_x = 0

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[layer_x:] = EPS

    # Магнитная проницаемость
    mu = np.ones(maxSize)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize)
    source = GaussianDiffPlaneWave(30.0, 10.0, Sc, eps[sourcePos], mu[sourcePos])

    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]

    # Расчет коэффициентов для граничного условия слева
    tempLeft = Sc / np.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize, -1.1, 1.1, 'Ez, В/м')
    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    display.drawBoundaries(layer_x)

    Ez_lst = list()

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy[:-1] = Hy[:-1] + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu[:-1])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:] = Ez[1:] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps[1:]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        # Граничные условия ABC первой степени
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 2 == 0:
            Ez_lst.append(Ez.copy())

    # Путь к папке с результатами
    dir_ = pathlib.Path("results")
    # Создаем папку, если она не существует
    dir_.mkdir(exist_ok=True)
    # Запуск анимации
    ani = display.start_animation(Ez_lst)
    # Сохранение анимации
    ani.save("results/task3.gif")

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(
        probes, -1.1, 1.1, filename="results/task3_probeSignals.png"
    )
    tools.show_signal_spectrum(
        probes, delta_t, filename="results/task3_signalSpectrum.png"
    )
