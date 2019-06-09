import json
import matplotlib.pyplot as plt
import numpy as np
import imageio as sc
import sys
import glob

sys.path.insert(0, '../includes/')

import helper
import plot_config

params = {
    'report_files': 'mc/mountain-car-paper-demo-3/openaigym.episode_batch.0.*.stats.json',
    'env_image': 'mountain_car.png',
    'episodes': 15,
    'xlim': [0, 15],
    'ylim': [-6500, 100],
    'xlabel': 'Episode',
    'ylabel': 'Reward',
    'color': plot_config.custom_colors['intense sandy brown'],
    'color_dqn': plot_config.custom_colors['japanese indigo'],
    'window_size': 3,
    'figname': 'reward_mountain_car.{ext}',
}


def parse_reward_file(fn):
    with open(fn, 'r') as f:
        report = json.load(f)

    assert len(report['episode_rewards']) >= params['episodes'], 'too few episodes recorded'
    return report['episode_rewards']


def load_rewards(report_files):
    fns = glob.glob(report_files)
    if not fns: raise FileNotFoundError('no report files found')

    rewards = []
    for i, fn in enumerate(fns):
        rewards.append(parse_reward_file(fn)[:params['episodes']])

    return np.mean(rewards, axis=0), np.std(rewards, axis=0)


def plot_reward(params):
    fig = plt.figure(figsize=plot_config.double_figure_size)

    ax_rew = fig.add_axes([0.23, 0.22, 0.74, 0.77])
    ax_rew.set_xlabel(params['xlabel'], fontsize=plot_config.fontsize_regular)
    ax_rew.set_ylabel(params['ylabel'], fontsize=plot_config.fontsize_regular)
    ax_rew.set_xlim(params['xlim'])
    ax_rew.set_ylim(params['ylim'])
    plot_config.remove_top_and_right_spines(ax_rew)
    plot_config.set_tick_fontsize(ax_rew, plot_config.fontsize_small)

    ax_env = fig.add_axes([0.58, 0.3, 0.37, 0.37])
    ax_env.set_xticks([])
    ax_env.set_yticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax_env.spines[axis].set_linewidth(0.5)

    ax_env.imshow(sc.imread(params['env_image']), interpolation='none')

    # ax_rew.plot(rewards[-1], label=i)

    mean_rewards, std_rewards = load_rewards(params['report_files'])
    #mean_rewards_dqn, std_rewards_dqn = load_rewards(params['report_files_dqn'])

    ax_rew.axhline(-110, color=plot_config.custom_colors['light gray'], alpha=0.7, label='solved', lw=1.5, zorder=-1)

    #ax_rew.fill_between(np.arange(params['episodes']), mean_rewards_dqn - std_rewards_dqn, mean_rewards_dqn + std_rewards_dqn, color=params['color_dqn'], alpha=0.5, linewidth=0)
    #ax_rew.plot(np.arange(params['episodes']), mean_rewards_dqn, lw=1.5, color=params['color_dqn'], label='Q-learning')

    ax_rew.fill_between(np.arange(params['episodes']), mean_rewards - std_rewards, mean_rewards + std_rewards, color=params['color'], alpha=0.5, linewidth=0)
    ax_rew.plot(np.arange(params['episodes']), mean_rewards, lw=1.5, color=params['color'], label='Our model')

    # ax_rew.legend(loc=(0.1, 0.1), fontsize=plot_config.fontsize_tiny)

    print('[created]', params['figname'])
    fig.savefig(params['figname'].format(ext='pdf'))
    fig.savefig(params['figname'].format(ext='svg'))


plot_reward(params)
