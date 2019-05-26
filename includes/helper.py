import numpy as np
import svgutils.compose as sc
import plot_config


def sliding_window(data, window_size):
    return [None] + [np.mean(data[i - window_size // 2:i + window_size // 2 + 1]) for i in range(window_size // 2, len(data))]


def panel4x4(fn0, fn1, fn2, fn3, fn_out):
    single_size = plot_config.single_figure_size
    sc.Figure(str(single_size[0] * 150) + "px", str(single_size[1] * 150) + "px",
              sc.Panel(sc.SVG(fn0).scale(1.0).move(0, 0)),
              sc.Text("A", 5, 25, size=plot_config.fontsize_xhuge, weight='bold'),
              sc.Panel(sc.SVG(fn1).scale(1.0).move(
                  single_size[0] * 150 / 2., 0)),
              sc.Text("B", single_size[0] * 150 / 2 + 5., 25,
                      size=plot_config.fontsize_xhuge, weight='bold'),
              sc.Panel(sc.SVG(fn2).scale(1.0).move(
                  0, single_size[1] * 150 / 2.)),
              sc.Text("C", 5, single_size[
                      1] * 150 / 2. + 25, size=plot_config.fontsize_xhuge, weight='bold'),
              sc.Panel(sc.SVG(fn3).scale(1.0).move(
                  single_size[0] * 150 / 2., single_size[1] * 150 / 2.)),
              sc.Text("D", single_size[0] * 150 / 2. + 5, single_size[1] *
                      150 / 2. + 25, size=plot_config.fontsize_xhuge, weight='bold'),
              ).save(fn_out)


def panel2x2(fn0, fn1, fn_out, single_size=plot_config.single_figure_size):
    sc.Figure(str(single_size[0] * 150) + "px", str(single_size[1] * 80) + "px",
              sc.Panel(sc.SVG(fn0).scale(1.0).move(0, 15)),
              sc.Text("A", 4, 12, size=plot_config.fontsize_large, weight='bold', font='serif'),
              sc.Panel(sc.SVG(fn1).scale(1.0).move(
                  single_size[0] * 150 / 2., 15)),
              sc.Text("B", single_size[0] * 150 / 2 + 4., 12,
                      size=plot_config.fontsize_large, weight='bold', font='serif')
              ).save(fn_out)


def get_weights_data(fn, sources=None, targets=None):
    data = np.loadtxt(fn)
    senders = data[:, 0].astype(np.int)
    receivers = data[:, 1].astype(np.int)
    times = data[:, 2]
    weights = data[:, 3]

    if sources is None:
        sources = sorted(np.unique(senders))
    if targets is None:
        targets = sorted(np.unique(receivers))

    n_times = len(np.unique(times))
    n_sources = len(sources)
    n_targets = len(targets)

    time_data = np.zeros(n_times)
    weight_data = np.zeros((n_times, n_sources, n_targets))

    delta_t = np.diff(np.unique(times))[0]
    for i in range(len(times)):
        if senders[i] in sources and receivers[i] in targets:
            time_data[int(times[i] / delta_t) - 1] = times[i]
            weight_data[int(times[i] / delta_t) - 1][senders[i] -
                                                     np.min(sources)][receivers[i] - np.min(targets)] = weights[i]

    return time_data, sources, targets, weight_data
