#!/bin/env python3
import pathlib
from math import sin, sqrt

import matplotlib.pyplot as plt

A = 512
start, stop, step = -512, 512, 1
f = lambda x: -(A + 47) * sin(sqrt(abs(x / 2 + (A + 47)))) - x * sin(
    sqrt(abs(x - (A + 47)))
)

x = [i for i in range(start, stop + 1, step)]
y = [f(i) for i in x]

res = pathlib.Path("results")
res.mkdir(exist_ok=True)
file = res / "task1.txt"

with file.open("w") as f:
    for a, b in zip(x, y):
        f.write(f"{a}    {b}\n")

plt.plot(x, y)
plt.grid()
plt.savefig("results/task1.png")
