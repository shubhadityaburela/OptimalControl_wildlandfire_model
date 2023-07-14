import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import moviepy.video.io.ImageSequenceClip
import glob
import jax.numpy as jnp

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=200, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


class PlotFlow:
    def __init__(self, X, Y, t) -> None:

        self.Nx = int(np.size(X))
        self.Ny = int(np.size(Y))
        self.Nt = int(np.size(t))

        self.X = X
        self.Y = Y
        self.t = t

        # Prepare the space-time grid for 1D plots
        self.X_1D_grid, self.t_grid = np.meshgrid(X, t)
        self.X_1D_grid = self.X_1D_grid.T
        self.t_grid = self.t_grid.T

        # Prepare the space grid for 2D plots
        self.X_2D_grid, self.Y_2D_grid = np.meshgrid(X, Y)
        self.X_2D_grid = np.transpose(self.X_2D_grid)
        self.Y_2D_grid = np.transpose(self.Y_2D_grid)

    def plot1D(self, Q):
        immpath = "./plots/FOM_1D/primal/"
        os.makedirs(immpath, exist_ok=True)

        T = Q[:self.Nx, :]
        S = Q[self.Nx:, :]

        # Plot the snapshot matrix for conserved variables for original model
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        im1 = ax1.pcolormesh(self.X_1D_grid, self.t_grid, T, cmap='YlOrRd')  # , vmin=0, vmax=jnp.max(T))
        ax1.axis('off')
        # ax1.axis('scaled')
        ax1.set_title(r"$T(x, t)$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2 = fig.add_subplot(122)
        im2 = ax2.pcolormesh(self.X_1D_grid, self.t_grid, S, cmap='YlGn')
        ax2.axis('off')
        # ax2.axis('scaled')
        ax2.set_title(r"$S(x, t)$")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        fig.supylabel(r"time $t$")
        fig.supxlabel(r"space $x$")

        save_fig(filepath=immpath + 'Var', figure=fig)

    def plot2D(self, Q, save_plot=False, plot_every=10, plot_at_all=False):
        Q = np.reshape(np.transpose(Q), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")

        if plot_at_all:
            if not save_plot:
                plt.ion()
                fig, ax = plt.subplots(1, 2)
                for n in range(self.Nt):
                    if n % plot_every == 0:
                        min_T = np.min(Q[n, 0, :, :])
                        max_T = np.max(Q[n, 0, :, :])
                        min_S = np.min(Q[n, 1, :, :])
                        max_S = np.max(Q[n, 1, :, :])
                        ax[0].pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q[n, 0, :, :]), vmin=min_T, vmax=max_T, cmap='YlOrRd')
                        ax[0].axis('scaled')
                        ax[0].set_title("T")
                        ax[1].pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q[n, 1, :, :]), vmin=min_S, vmax=max_S, cmap='YlGn')
                        ax[1].axis('scaled')
                        ax[1].set_title("S")

                        fig.supylabel(r"$Y$")
                        fig.supxlabel(r"$X$")

                        plt.draw()
                        plt.pause(0.5)
                        ax[0].cla()
                        ax[1].cla()
            else:
                immpath = "./plots/FOM_2D/primal/"
                os.makedirs(immpath, exist_ok=True)
                for n in range(self.Nt):
                    if n % plot_every == 0:
                        min_T = np.min(Q[n, 0, :, :])
                        max_T = np.max(Q[n, 0, :, :])
                        min_S = np.min(Q[n, 1, :, :])
                        max_S = np.max(Q[n, 1, :, :])

                        fig = plt.figure(figsize=(10, 5))
                        ax1 = fig.add_subplot(121)
                        im1 = ax1.pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q[n, 0, :, :]), vmin=min_T, vmax=max_T, cmap='YlOrRd')
                        ax1.axis('scaled')
                        ax1.set_title(r"$T(x, y)$")
                        ax1.set_yticks([], [])
                        ax1.set_xticks([], [])
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='10%', pad=0.08)
                        fig.colorbar(im1, cax=cax, orientation='vertical')

                        ax2 = fig.add_subplot(122)
                        im2 = ax2.pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q[n, 1, :, :]), vmin=min_S, vmax=max_S, cmap='YlGn')
                        ax2.axis('scaled')
                        ax2.set_title(r"$S(x, y)$")
                        ax2.set_yticks([], [])
                        ax2.set_xticks([], [])
                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='10%', pad=0.08)
                        fig.colorbar(im2, cax=cax, orientation='vertical')

                        fig.supylabel(r"space $y$")
                        fig.supxlabel(r"space $x$")

                        fig.savefig(immpath + "Var" + str(n), dpi=300, transparent=True)
                        fig.savefig(immpath + "Var" + str(n), format="pdf", bbox_inches="tight", transparent=True)
                        plt.close(fig)

                fps = 1
                image_files = sorted(glob.glob(os.path.join(immpath, "*.png")), key=os.path.getmtime)
                clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                clip.write_videofile(immpath + "Var" + '.mp4')
        else:
            pass
