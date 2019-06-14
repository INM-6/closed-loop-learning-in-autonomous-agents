import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as sc
import sys
import os
import glob

sys.path.insert(0, '../includes/')

import helper
import plot_config

params = {
    'report_files': 'fl/report.json',
    'labels': ['network performance'],
    'xlabel': 'Steps',
    'ylabel': 'Reward (500 steps)',
    'window_size': 500,
    'color_reward': plot_config.custom_colors['sandy brown'],
    'env_image': 'frozen_lake.png',
    'in_file': 'in-41-0.dat',
    'actor_file': 'actor-42-0.dat',
    'critic_file': 'critic-43-0.dat',
    'reward_file': 'reward-44-0.dat',
    'wr_file': 'weight_recorder-45-0.csv',
    'xlabel_value_policy': 'Position $x$',
    'ylabel_value_policy': 'Position $y$',
    'xticks_value_policy': np.linspace(0, 3, 4),
    'yticks_value_policy': np.linspace(0, 3, 4),
    'actor_scaling_value_policy': 2.0,
    'figname_critic': 'critic_frozen_lake.svg',
    'figname_reward': 'reward_frozen_lake.svg',
    'figname_value_policy': 'value_policy_frozen_lake.svg',
    'figname_env': 'frozen_lake_env.svg',
    'figname': 'frozen_lake.svg',
}


def plot_weights(params):

    fig = plt.figure(figsize=plot_config.single_figure_size)
    fig.subplots_adjust(0.15, 0.15, 0.9, 0.9)
    ax = fig.add_subplot(111)
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    input_neurons = np.unique(np.loadtxt(params['in_file'])[:, 0]).astype(np.int)
    critic_neurons = np.unique(np.loadtxt(params['critic_file'])[:, 0]).astype(np.int)

    times, _, _, weights = helper.get_weights_data(params['wr_file'], input_neurons, critic_neurons)

    # plot weight course for each synapse
    ax.set_xlabel('time (s)')
    ax.set_ylabel('value')
    for i in input_neurons:
        for k, j in enumerate(critic_neurons):
            plt.plot(times / 1000., weights[:, i - np.min(input_neurons), j - np.min(critic_neurons)], label='pre: {}({}), post: {}'.format(i, i - np.min(input_neurons), j), color=list(plot_config.custom_colors.values())[(i+i*k)%7])
    fig.savefig( params["figname_critic"])



def parse_reward(fn):
    episodes = []
    rewards = []
    with open(fn, 'r') as f:
        report = json.load(f)
    for j, key in enumerate(sorted(report, key=int)):
        episodes.append(key)
        rewards += report[key]['reward']
    return episodes, rewards

def plot_reward(params):
    fig = plt.figure(figsize=plot_config.double_figure_size)
    ax = fig.add_axes([0.23, 0.2, 0.74, 0.77])
    ax.set_xlabel(params['xlabel'], fontsize=plot_config.fontsize_regular)
    ax.set_ylabel(params['ylabel'], fontsize=plot_config.fontsize_regular)
    ax.set_xlim([0, 4000])
    ax.set_xticks(np.arange(0, 5000, 1000))
    plot_config.remove_top_and_right_spines(ax)
    plot_config.set_tick_fontsize(ax, plot_config.fontsize_small)

    ax_env = fig.add_axes([0.65, 0.3, 0.35, 0.3])
    ax_env.set_xticks([])
    ax_env.set_yticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax_env.spines[axis].set_linewidth(0)

    slides = []
    fns = glob.glob(params['report_files'])
    for i, fn in enumerate(fns):
        episodes, rewards = parse_reward(fn)
        reward_slide = helper.sliding_window(rewards, params['window_size'])
        slides.append(reward_slide)


    min_length = min([len(x) for x in slides])
    for i, reward_slide in enumerate(slides):
        slides[i] = np.array(reward_slide[1:min_length])
        col = params['color_reward']


    slides = np.array(slides)

    mean_slide = np.mean(slides, axis=0)
    std_slide = np.std(slides, axis=0)
    col = params['color_reward']
    ax.plot(mean_slide, '-', color=col, lw=1.5, alpha=1.0, label=params["labels"][0])
    ax.fill_between(np.arange(len(mean_slide)), mean_slide - std_slide, mean_slide + std_slide, color=col, alpha=0.5, linewidth=0)

    ax.axhline(1. / 6., color=plot_config.custom_colors['light gray'], label='theoretical optimum', alpha=0.7, lw=1.5, zorder=-1)
    ax.set_ylim([-0.15, 0.2])

    ax_env.imshow(sc.imread(params['env_image']), interpolation='none')

    print(params['figname_reward'])
    fig.savefig(params['figname_reward'])


def plot_value_policy_maps(params):
    d_actor_arrows = {
        0: np.array([-1., 0]),
        1: np.array([0., -1.]),
        2: np.array([1., 0.]),
        3: np.array([0., 1.]),
    }

    input_neurons = np.unique(np.loadtxt(params['in_file'])[:, 0]).astype(np.int)
    actor_neurons = np.unique(np.loadtxt(params['actor_file'])[:, 0]).astype(np.int)
    critic_neurons = np.unique(np.loadtxt(params['critic_file'])[:, 0]).astype(np.int)

    n_input_neurons = len(input_neurons)
    n_actor_neurons = len(actor_neurons)

    _, _, _, weights_critic = helper.get_weights_data(params['wr_file'], input_neurons, critic_neurons)
    _, _, _, weights_actor = helper.get_weights_data(params['wr_file'], input_neurons, actor_neurons)

    vector_actor_begin = []
    vector_actor_end = []
    for i in range(n_input_neurons):
        vec_begin = []
        vec_end = []
        for j in range(n_actor_neurons):
            vec_begin.append(weights_actor[0][i][j] * d_actor_arrows[j])
            vec_end.append(weights_actor[-1][i][j] * d_actor_arrows[j])
        vector_actor_begin.append(vec_begin)
        vector_actor_end.append(vec_end)

    sqrt_num_input_neurons = int(np.sqrt(n_input_neurons))

    xmin = min(params['xticks_value_policy'])
    xmax = max(params['xticks_value_policy'])
    ymin = min(params['yticks_value_policy'])
    ymax = max(params['yticks_value_policy'])

    fig = plt.figure(figsize=plot_config.double_figure_size)

    xdiff = np.diff(params['xticks_value_policy'])[0]
    xticks = np.arange(xmin, xmax + 1.5 * xdiff, xdiff) - xdiff / 2.

    ydiff = np.diff(params['yticks_value_policy'])[0]
    yticks = np.arange(ymin, ymax + 1.5 * ydiff, ydiff) - ydiff / 2.

    ax_post = fig.add_axes([0.2, 0.2, 0.74, 0.77])
    ax_post.set_xlabel(params['xlabel_value_policy'], fontsize=plot_config.fontsize_regular)
    ax_post.set_ylabel(params['ylabel_value_policy'], fontsize=plot_config.fontsize_regular)
    ax_post.set_xlim([xmin - xdiff / 2., xmax + xdiff / 2.])
    ax_post.set_ylim([ymin - ydiff / 2., ymax + ydiff / 2.])
    plot_config.set_tick_fontsize(ax_post, plot_config.fontsize_small)

    map_post = ax_post.pcolormesh(xticks, yticks, weights_critic[-1].reshape(sqrt_num_input_neurons, sqrt_num_input_neurons)[::-1], cmap=plot_config.cmap, vmin=-.6, vmax=.6)
    for i in range(n_input_neurons):
        x_idx = int(i % sqrt_num_input_neurons)
        y_idx = int(sqrt_num_input_neurons - (i / sqrt_num_input_neurons) - 1)

        ax_post.arrow(params['xticks_value_policy'][x_idx], params['yticks_value_policy'][y_idx],
                      params['actor_scaling_value_policy'] * np.mean(vector_actor_end[i], axis=0)[0],
                      params['actor_scaling_value_policy'] * np.mean(vector_actor_end[i], axis=0)[1],
                      fc='k', ec='k', lw=1, width=0.003)

    ax_post.set_xticks(params['xticks_value_policy'])
    ax_post.set_yticks(params['yticks_value_policy'])
    ax_post.set_xticklabels([int(i) for i in params['xticks_value_policy']])
    ax_post.set_yticklabels([int(i) for i in params['yticks_value_policy']][::-1])

    cb = plt.colorbar(map_post, ax=[ax_post])
    cb.set_ticks([-0.5, 0.0, 0.5])
    cb.set_ticklabels([-0.5, 0.0, 0.5], plot_config.fontsize_small)

    print(params['figname_value_policy'])
    fig.savefig(params['figname_value_policy'])


def create_4x4_panel_plot():
    helper.panel4x4(params['figname_env'], params['figname_reward'], params['figname_value_policy'], params['figname_critic'], params['figname'])


def create_2x2_panel_plot():
    print('[created]', params['figname'])
    helper.panel2x2(params['figname_reward'], params['figname_value_policy'], params['figname'], plot_config.double_figure_size)

plot_reward(params)
plot_weights(params)
plot_value_policy_maps(params)
create_2x2_panel_plot()
