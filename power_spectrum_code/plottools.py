import numpy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from generaltools import from_eta_to_k_par, from_u_to_k_perp, from_jansky_to_milikelvin


def colorbar(mappable, extend="neither"):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, extend=extend)


def plot_power_spectrum(
    u_bins,
    eta_bins,
    nu,
    data,
    norm=None,
    title=None,
    axes=None,
    colormap="viridis",
    axes_label_font=20,
    tickfontsize=15,
    xlabel_show=False,
    ylabel_show=False,
    zlabel_show=False,
    z_label=None,
    return_norm=False,
    colorbar_show=False,
    colorbar_limits="neither",
    ratio=False,
    diff=False,
    x_range=None,
    y_range=None,
):

    central_frequency = nu[int(len(nu) / 2)]
    x_values = from_u_to_k_perp(u_bins, central_frequency)
    y_values = from_eta_to_k_par(eta_bins, central_frequency)

    if ratio:
        z_values = data
    else:
        z_values = from_jansky_to_milikelvin(data, nu)

    x_label = r"$k_{\perp}$ [$h$Mpc$^{-1}$]"
    y_label = r"$k_{\parallel}$ [$h$Mpc$^{-1}$]"
    if z_label is None:
        z_label = r"Variance [mK$^2$ $h^{-3}$Mpc$^3$ ]"
    elif z_label == False:
        z_label = None

    if x_range is None:
        axes.set_xlim(9e-3, 3e-1)
    else:
        axes.set_xlim(x_range[0], x_range[1])
    if y_range is None:
        axes.set_ylim(9e-3, 1.2e0)
    else:
        axes.set_ylim(y_range[0], y_range[1])

    if diff:
        pass
    else:
        z_values[data < 0] = numpy.nan
    if norm is None:
        norm = colors.LogNorm(
            vmin=numpy.real(z_values).min(), vmax=numpy.real(z_values).max()
        )

    if title is not None:
        axes.set_title(title)
    psplot = axes.pcolor(
        x_values, y_values, z_values.T, cmap=colormap, rasterized=True, norm=norm
    )
    if colorbar_show:
        cax = colorbar(psplot, extend=colorbar_limits)
        cax.ax.tick_params(axis="both", which="major", labelsize=tickfontsize)
        cax.set_label(z_label, fontsize=axes_label_font)

    axes.set_xscale("log")
    axes.set_yscale("log")

    if xlabel_show:
        axes.set_xlabel(x_label, fontsize=axes_label_font)
    if ylabel_show:
        axes.set_ylabel(y_label, fontsize=axes_label_font)

    axes.tick_params(axis="both", which="major", labelsize=tickfontsize)

    return norm if return_norm else None
