import json
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, '../includes/')

import helper
import plot_config

params = {
    'in_file': 'new-paper-demo-in-58-0.dat',
    'actor_file': 'new-paper-demo-actor-59-0.dat',
    'critic_file': 'new-paper-demo-critic-60-0.dat',
    'reward_file': 'new-paper-demo-reward-61-0.dat',
    'xlabel': 'Time (s)',
    'ylabel': 'Activity',
    'min_time_initial': 12500.,
    'max_time_initial': 19000.,
    'min_time_final': 362500.,
    'max_time_final': 370000.,
    'figname': 'traces_mountain_car.{ext}',
}


def parse_data(params, fn):
    data = np.loadtxt(fn)
    senders = data[:, 0]
    times = data[:, 1]
    rates = data[:, 2]

    times_data = np.sort(np.unique(times))
    senders_data = np.sort(np.unique(senders))
    rates_data = []

    for i, s in enumerate(senders_data):
        rates_data.append(rates[np.where(senders == s)])

    return times_data, senders_data, np.array(rates_data)


def plot_traces(params):
    times_in, senders_in, rates_in = parse_data(params, params['in_file'])
    times_actor, senders_actor, rates_actor = parse_data(params, params['actor_file'])
    times_critic, senders_critic, rates_critic = parse_data(params, params['critic_file'])
    times_reward, senders_reward, rates_reward = parse_data(params, params['reward_file'])

    fig = plt.figure(figsize=plot_config.double_figure_size)

    ax_initial = fig.add_axes([0.23, 0.22, 0.33, 0.76])
    ax_initial.set_xlabel(params['xlabel'], x=1.1, fontsize=plot_config.fontsize_regular)
    # ax_initial.set_ylabel(params['ylabel'])
    ax_initial.set_xlim([params['min_time_initial'], params['max_time_initial'] + 150.])
    # ax_initial.set_xticks([10000, 12500, 15000])
    # ax_initial.set_xticklabels([10.0, 12.5, 15.0])
    ax_initial.set_xticks([12500, 15000, 17500])
    ax_initial.set_xticklabels([12.5, 15.0, 17.5])
    ax_initial.set_ylim([0., 6.])
    ax_initial.set_yticks([0.5, 2.2, 4., 5.5])
    ax_initial.set_yticklabels(['Place\n cells', 'Actor\n units', 'Critic', 'Prediction\n error'], fontsize=plot_config.fontsize_regular)
    plot_config.remove_top_and_right_spines(ax_initial)
    plot_config.set_tick_fontsize(ax_initial, plot_config.fontsize_small)
    ax_initial.yaxis.set_tick_params(width=0)

    ax_final = fig.add_axes([0.61, 0.22, 0.32, 0.76])
    ax_final.set_xlim([params['min_time_final'] - 200., params['max_time_final']])
    ax_final.set_xticks([362500, 367000, 370000])
    ax_final.set_xticklabels([362.5, 367.0, 370.0])
    ax_final.set_ylim([0., 6.])
    ax_final.spines['top'].set_visible(False)
    ax_final.spines['right'].set_visible(False)
    ax_final.spines['left'].set_visible(False)
    ax_final.get_xaxis().tick_bottom()
    ax_final.set_yticks([])
    plot_config.set_tick_fontsize(ax_final, plot_config.fontsize_small)

    ax_training = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])
    ax_training.set_xlim([0, 1])
    ax_training.set_ylim([0, 1])
    ax_training.add_patch(plt.Rectangle((0.56, 0.2), 0.05, 0.78, linewidth=0, color=plot_config.custom_colors['light gray'], alpha=0.5))
    ax_training.text(0.575, 0.63, '{}s'.format((params['min_time_final'] - params['max_time_initial']) / 1000.), rotation=90, fontsize=0.9 * plot_config.fontsize_small)

    # plot the diagonals indicating cut axis
    d = .005  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax_initial.transAxes, color='k', clip_on=False)
    ax_initial.plot((1 - d, 1 + d), (-3 * d, 3 * d), **kwargs)

    kwargs.update(transform=ax_final.transAxes)  # switch to the right axes
    ax_final.plot((-d, +d), (-3 * d, 3 * d), **kwargs)  # bottom-left diagonal

    pos_initial_in = np.where(np.logical_and(times_in > params['min_time_initial'], times_in < params['max_time_initial']))
    pos_final_in = np.where(np.logical_and(times_in > params['min_time_final'], times_in < params['max_time_final']))

    for i, s in enumerate(senders_in):
        ax_initial.plot(times_in[pos_initial_in], rates_in[i][pos_initial_in])
        ax_final.plot(times_in[pos_final_in], rates_in[i][pos_final_in])

    for i, s in enumerate(senders_actor):
        ax_initial.plot(times_actor[pos_initial_in], 0.5 * rates_actor[i][pos_initial_in] + 1.8)
        ax_final.plot(times_actor[pos_final_in], 0.5 * rates_actor[i][pos_final_in] + 1.8)

    ax_initial.plot(times_critic[pos_initial_in], rates_critic[0][pos_initial_in] + 4.)
    ax_final.plot(times_critic[pos_final_in], rates_critic[0][pos_final_in] + 4.)

    ax_initial.plot(times_reward[pos_initial_in], 10. * rates_reward[0][pos_initial_in] + 5.5)
    ax_final.plot(times_reward[pos_final_in], 10. * rates_reward[0][pos_final_in] + 5.5)

    print('[created]', params['figname'])
    fig.savefig(params['figname'].format(ext='svg'))
    fig.savefig(params['figname'].format(ext='pdf'))

plot_traces(params)
