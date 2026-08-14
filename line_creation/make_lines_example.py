import numpy as np
import matplotlib.pyplot as plt
from make_lines import LineProfileGenerator
from matplotlib import cm
from matplotlib.colors import ListedColormap
import line_analysis as la

line = 'Fe5250'
gen = LineProfileGenerator(line)

# n_instances = 100
# deg = 50
# gen.make_test_plot(deg, n_instances)

# Center (deg=0, mu=1.0) and Limb (deg=78.46, mu=0.2)
for mu in [1.0, 0.8, 0.5, 0.2]:
    deg = np.degrees(np.arccos(mu))
    
    # Evaluate DISCO profile (using median filling factors, i.e., quantiles = 0.5)
    prof, ffs = gen.get_full_profile(deg, GT_quantile=0.5, ratio_quantile=0.5)
    
    plt.plot(gen.wl, prof, label = mu)
plt.legend()
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title("DISCO")
plt.savefig("C2L")