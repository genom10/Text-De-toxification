import seaborn as sns
import matplotlib.pyplot as plt
def save_kde_reg_plot(df, x, y, save_path=None):
    g = sns.JointGrid()
    x, y = df[x], df[y]
    sns.histplot(x=x, ax=g.ax_marg_x)
    sns.histplot(y=y, ax=g.ax_marg_y)
    sns.kdeplot(x=x, y=y, ax=g.ax_joint)
    sns.regplot(x=x, y=y, scatter=False, ax=g.ax_joint, order=3)
    if save_path is not None:
        plt.savefig(save_path)