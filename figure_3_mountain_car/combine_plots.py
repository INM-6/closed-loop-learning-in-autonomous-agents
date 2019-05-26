import sys

sys.path.insert(0, '../includes/')

import helper
import plot_config


def create_2x2_panel_plot(params):
    print('[created]', params['figname'])
    helper.panel2x2(params['figname_reward'], params['figname_traces'], params['figname'], plot_config.double_figure_size)


params = {
    'figname_reward': '../../manuscript/figures/reward_mountain_car.svg',
    'figname_traces': '../../manuscript/figures/traces_mountain_car.svg',
    'figname': '../../manuscript/figures/mountain_car.svg',
}

create_2x2_panel_plot(params)
