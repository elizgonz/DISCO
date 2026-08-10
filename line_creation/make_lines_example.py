import numpy as np
import matplotlib.pyplot as plt
from make_lines import LineProfileGenerator
from matplotlib import cm
from matplotlib.colors import ListedColormap
import line_analysis as la

line = 'Fe6173'
lpg = LineProfileGenerator(line)
n_instances = 100
deg = 50

lpg.make_test_plot(deg, n_instances)