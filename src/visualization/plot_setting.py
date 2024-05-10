import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import os

# Setting the style
def setup_plotting():
    colors = cycler(color=sns.color_palette('Set2'))
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "lightgray"
    plt.rcParams["axes.prop_cycle"] = colors
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.titlesize"] = 25
    plt.rcParams["figure.dpi"] = 100

# Utility function to save plots
def save_plot(ax, title, save=False):
    if save:
        safe_title = title.replace(' ', '_').replace(':', '').replace(',', '').replace('/', '').replace('\\', '')
        filename = f"{safe_title}.png"
        
        # Define the path to save the figure
        figures_path = '../../reports/figures/'
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)
        
        full_path = os.path.join(figures_path, filename)
        
        # Save the plot
        ax.figure.savefig(full_path)


