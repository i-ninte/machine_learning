from statsmodels.stats.power import TTestIndPower
import numpy as np
import matplotlib.pyplot as plt

# Calculate power curves from multiple power analyses
analysis = TTestIndPower()

power_plot = analysis.plot_power(dep_var='nobs', nobs=np.arange(5, 100), effect_size=np.array([0.2, 0.5, 0.8]))

# Save the plot as an image file
power_plot.figure.savefig('power_curves.png')

# Optionally, you can also display the saved image using an image viewer
plt.show()
