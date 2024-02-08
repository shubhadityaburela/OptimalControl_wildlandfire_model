import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import moviepy.video.io.ImageSequenceClip
import glob

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

    def plot1D(self, Q, name, immpath):
        os.makedirs(immpath, exist_ok=True)

        # Plot the snapshot matrix for conserved variables for original model
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        im1 = ax1.pcolormesh(self.X_1D_grid, self.t_grid, Q, cmap='YlOrRd')
        ax1.axis('off')
        ax1.set_title(r"$q(x, t)$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        fig.supylabel(r"time $t$")
        fig.supxlabel(r"space $x$")

        fig.savefig(immpath + name, dpi=300, transparent=True)

    def plot1D_ROM_converg(self, J, Jf, dL_du, itr, Nm, immpath):

        x = np.arange(len(J))

        os.makedirs(immpath, exist_ok=True)

        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, label="1")
        ax1.semilogy(np.arange(len(J)), J, color="C0", label="Cost functional")
        ax1.semilogy(np.arange(len(Jf)), Jf, color="C2", label="Cost functional (FOM)")
        ax1.set_xlabel(r"$n_{\mathrm{iter}}$", color="C0")
        ax1.set_ylabel(r"$J$", color="C0")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.tick_params(axis='x', colors="C0")
        ax1.tick_params(axis='y', colors="C0")
        ax1.plot(itr, [J[i] for i in itr], ls="", marker="*", color="C1", label="basis refinement steps")
        ax1.legend()
        fig1.savefig(immpath + "J", dpi=300, transparent=True)

        fig3 = plt.figure(figsize=(8, 8))
        ax3 = fig3.add_subplot(111, label="4")
        ax3.semilogy(np.arange(len(dL_du)), dL_du)
        ax3.set_xlabel(r"$n_{\mathrm{iter}}$", color="C0")
        ax3.set_ylabel(r"relative $\quad \frac{dL}{du}$", color="C0")
        fig3.savefig(immpath + "dL_du_ratio", dpi=300, transparent=True)

        fig4 = plt.figure(figsize=(8, 8))
        ax4 = fig4.add_subplot(111, label="5")
        ax4.plot(np.arange(len(Nm)), Nm, label="Truncated modes")
        ax4.set_xlabel(r"$n_{\mathrm{refine}}$", color="C0")
        ax4.set_ylabel(r"$n_{\mathrm{modes}}$", color="C0")
        ax4.legend()
        fig4.savefig(immpath + "trunc_modes", dpi=300, transparent=True)




    def plot1D_FOM_converg(self, J, dL_du, immpath):

        x = np.arange(len(J))

        os.makedirs(immpath, exist_ok=True)

        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, label="1")
        ax1.semilogy(np.arange(len(J)), J, color="C0", label="Cost functional")
        ax1.set_xlabel(r"$n_{\mathrm{iter}}$", color="C0")
        ax1.set_ylabel(r"$J$", color="C0")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.tick_params(axis='x', colors="C0")
        ax1.tick_params(axis='y', colors="C0")
        fig1.savefig(immpath + "J", dpi=300, transparent=True)

        fig3 = plt.figure(figsize=(8, 8))
        ax3 = fig3.add_subplot(111, label="4")
        ax3.semilogy(np.arange(len(dL_du)), dL_du)
        ax3.set_xlabel(r"$n_{\mathrm{iter}}$", color="C0")
        ax3.set_ylabel(r"relative $\quad \frac{dL}{du}$", color="C0")
        fig3.savefig(immpath + "dL_du_ratio", dpi=300, transparent=True)

    def plot2D(self, Q, name, immpath, save_plot=False, plot_every=10, plot_at_all=False):
        Q_T = np.reshape(np.transpose(Q[:self.Nx * self.Ny]), newshape=[self.Nt, self.Nx, self.Ny], order="F")
        Q_S = np.reshape(np.transpose(Q[self.Nx * self.Ny:]), newshape=[self.Nt, self.Nx, self.Ny], order="F")

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
                        ax[0].pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q[n, 0, :, :]), vmin=min_T,
                                         vmax=max_T, cmap='YlOrRd')
                        ax[0].axis('scaled')
                        ax[0].set_title("T")
                        ax[1].pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q[n, 1, :, :]), vmin=min_S,
                                         vmax=max_S, cmap='YlGn')
                        ax[1].axis('scaled')
                        ax[1].set_title("S")

                        fig.supylabel(r"$Y$")
                        fig.supxlabel(r"$X$")

                        plt.draw()
                        plt.pause(0.5)
                        ax[0].cla()
                        ax[1].cla()
            else:
                os.makedirs(immpath, exist_ok=True)
                for n in range(self.Nt):
                    if n % plot_every == 0:
                        min_T = np.min(Q_T[n, :, :])
                        max_T = np.max(Q_T[n, :, :])
                        min_S = np.min(Q_S[n, :, :])
                        max_S = np.max(Q_S[n, :, :])

                        fig = plt.figure(figsize=(10, 5))
                        ax1 = fig.add_subplot(121)
                        im1 = ax1.pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q_T[n, :, :]), vmin=min_T,
                                             vmax=max_T, cmap='YlOrRd')
                        ax1.axis('scaled')
                        ax1.set_title(r"$T(x, y)$")
                        ax1.set_yticks([], [])
                        ax1.set_xticks([], [])
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='10%', pad=0.08)
                        fig.colorbar(im1, cax=cax, orientation='vertical')

                        ax2 = fig.add_subplot(122)
                        im2 = ax2.pcolormesh(self.X_2D_grid, self.Y_2D_grid, np.squeeze(Q_S[n, :, :]), vmin=min_S,
                                             vmax=max_S, cmap='YlGn')
                        ax2.axis('scaled')
                        ax2.set_title(r"$S(x, y)$")
                        ax2.set_yticks([], [])
                        ax2.set_xticks([], [])
                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='10%', pad=0.08)
                        fig.colorbar(im2, cax=cax, orientation='vertical')

                        fig.supylabel(r"space $y$")
                        fig.supxlabel(r"space $x$")

                        fig.savefig(immpath + name + str(n), dpi=300, transparent=True)
                        plt.close(fig)

                # fps = 1
                # image_files = sorted(glob.glob(os.path.join(immpath, "*.png")), key=os.path.getmtime)
                # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                # clip.write_videofile(immpath + name + '.mp4')
        else:
            pass
