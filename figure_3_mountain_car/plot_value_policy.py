import json
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, '../includes/')

import helper
import plot_config

params = {
    # 'weight_recorder_file': '../../data/mountain_car/experiment5_weight_recorder-62-0.csv',
    # 'report_file': '../../data/mountain_car/report_trial2.json',
    'weight_recorder_file': '../../data/mountain_car/new-paper-3_weight_recorder-62-0.csv',
    'report_file': '../../data/mountain_car/report_trial2.json',
    # 'weight_recorder_file': '../../data/mountain_car/trial2_weight_recorder-62-0.csv',
    # 'report_file': '../../data/mountain_car/report_trial2.json',
    'actor_arrows': {
        0: np.array([-1., 0]),
        1: np.array([0., 0.]),
        2: np.array([1., 0.]),
    },
    'num_inputs': 25,
    'gid_critic': [56],
    'gid_actor': [53, 54, 55],
    'vector_actor_scaling': 1.,
    'xlabel': 'Position $x$',
    'ylabel': 'Velocity $\dot{x}$',
    'figname': '../../data/figures/value_policy_mountain_car.svg',
}


def plot_value_policy(params):
    _, sources_critic, targets_critic, weights_critic = helper.get_weights_data(params['weight_recorder_file'], targets=params['gid_critic'])
    _, sources_actor, targets_actor, weights_actor = helper.get_weights_data(params['weight_recorder_file'], targets=params['gid_actor'])

    vector_actor = []
    for i in xrange(params['num_inputs']):
        vec = []
        for j in xrange(len(params['gid_actor'])):
            vec.append(weights_actor[-1][i][j] * params['actor_arrows'][j])
        vector_actor.append(np.mean(vec, axis=0))

    with open(params['report_file'], 'r') as f:
        report = json.load(f)

    rew = []
    obs = []
    for key in sorted(report, key=int):
        rew.append(np.sum(report[key]['reward']))
        obs.append(np.array(report[key]['obervation']))

        # need to scale observations to [0.5, 4.5], since grid is
        # defined over that range; could also rescale grid, but that
        # messes up actor arrows
        obs[-1][:, 0] = (obs[-1][:, 0] + 0.3) * 2. / 0.9 + 2.5
        obs[-1][:, 1] = obs[-1][:, 1] * 2. / 0.07 + 2.5

    fig = plt.figure(figsize=plot_config.single_figure_size)
    fig.subplots_adjust(0.17, 0.15, 0.9, 0.9)
    ax = fig.add_subplot(111)
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels([-1.2, -0.75, -0.3, 0.15, 0.6])
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels([-0.07, -0.035, 0., 0.035, 0.07])

    sqrt_num_inputs = int(np.sqrt(params['num_inputs']))
    value_map = plt.pcolormesh(weights_critic[-1].reshape(sqrt_num_inputs, sqrt_num_inputs), vmin=-.5, vmax=.5, cmap=plot_config.cmap)
    for i in xrange(params['num_inputs']):
        x_idx = i % sqrt_num_inputs
        y_idx = i / sqrt_num_inputs
        if np.dot(vector_actor[i], vector_actor[i]) > 1e-12:
            alpha = 1.
        else:
            alpha = 0.3
        ax.arrow(x_idx + 0.5, y_idx + 0.5,
                 params['vector_actor_scaling'] * vector_actor[i][0],
                 params['vector_actor_scaling'] * vector_actor[i][1],
                 fc='k', ec='k', alpha=alpha)

    # for ep in xrange(0, len(obs)):
    # for i, ep in enumerate([1, 2, 3, 14]):
    for i, ep in enumerate([np.argmax(rew)]):
        # ax.plot(obs[ep][:, 0], obs[ep][:, 1], marker='.', markersize=1., color=str(-80. / rew[ep]), lw=0.5, alpha=0.8)
        ax.plot(obs[ep][:, 0], obs[ep][:, 1], marker='.', markersize=2., color='0.9', lw=1, alpha=0.8)
        # ax.scatter(obs[ep][:, 0] + 0.02 * np.random.rand(len(obs[ep])), obs[ep][:, 1] + 0.02 * np.random.rand(len(obs[ep])), marker='.', c=np.linspace(0, 1, len(obs[ep])), lw=0.5, alpha=0.8, cmap='viridis', linewidth=0)

    # plt.colorbar(value_map, ax=[ax], label='Value')
    print params['figname']
    fig.savefig(params['figname'])

plot_value_policy(params)
