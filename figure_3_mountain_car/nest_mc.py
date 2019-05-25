#!/home/jordan/opt/miniconda3/envs/python2/bin/python
# !/usr/bin/python

import nest
from nest import raster_plot as rplt
import pylab as plt
import numpy as np
import sys
from datetime import datetime
from optparse import OptionParser
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

to_ms = lambda t: t * 1000.

opt_parser = OptionParser()
opt_parser.add_option('-t', '--simtime', dest='simtime',
                      type='float', help='Simulation time in s')
# opt_parser.add_option('-s', '--timestep', dest='music_timestep', type='float', help='MUSIC timestep')
opt_parser.add_option('-n', '--num_neurons_in', dest='num_neurons_in',
                      type='int', help='Number of encoding neurons')
opt_parser.add_option('-m', '--num_neurons_out', dest='num_neurons_out',
                      type='int', help='Number of decoding neurons')

(options, args) = opt_parser.parse_args()

#######################################
# Define parameters

params = {
    'dt': 1.,
    'delay': 1.0,
    'sigma': .0,
    'neuron_model': 'thresholdlin_rate_ipn',
}

input_params = {
    'tau': 1.,
    'g': 1.,
    'mean': 0.,
    'std': 0.0,
    'theta': -.5,
}

critic_params = {
    'tau': 5.,
    'mean': 0.0,
    'std': 0.0,
    'c': -1.,
    'theta': -1.,
}

# critic_params_exp = {
#     'tau': 5.,
#     'mean': 0.0,
#     'std': 0.0,
#     'c': -1.,
#     'theta': 0.,
# }

reward_params = {
    'tau': 5.,
    'mean': 0.0,
    # 'mean': -0.001,
    'std': 0.0,
    'theta': -0.999,
    # 'theta': -1.,
}

# reward_thresh_params = {
#     'tau': 0.01,
#     'mean': 0.0,
#     'std': 0.0,
# }

input_critic_params = {
    'A': 5 * 0.0025,
    # 'A': 0.,
    'weight_decay_constant': 0.0,
    'Wmax': 1.,
    'Wmin': -1.,
    'post_threshold': -1.,
}

critic_reward_params = {
    'tau_r': 20000.,
    'delay': 1.,
}

output_params = {
    'g': 1.,
    'tau': .5,
    'mean': 0.,
    'std': 0.05,
    'c': 0.,
}

input_actor_params = {
    'A': 1 * 0.025,
    'weight_decay_constant': 0.0,
    'Wmax': 1.,
    'Wmin': .05,
    'post_threshold': 0.1,
}

# alpha = 1.3
# beta_1 = -2.
# beta_2 = 0.25

NUM_ENC_NEURONS = options.num_neurons_in
NUM_DEC_NEURONS = options.num_neurons_out
NUM_REWARD_NEURONS = 1

#######################################
# Initialize Kernel

# np.random.seed(int(time.time()))
np.random.seed(123)

nest.ResetKernel()
nest.set_verbosity('M_FATAL')
nest.SetKernelStatus({'resolution': params['dt'], 'print_time': True,
                      'use_wfr': False, 'overwrite_files': True})

#######################################
# Create MUSIC devices

reward_in = nest.Create('music_rate_in_proxy', NUM_REWARD_NEURONS)
nest.SetStatus(reward_in, [{'port_name': 'reward_in',
                            'music_channel': c} for c in range(NUM_REWARD_NEURONS)])
nest.SetAcceptableLatency('reward_in', params['delay'])  # useless?


proxy_in = nest.Create('music_rate_in_proxy', NUM_ENC_NEURONS)
nest.SetStatus(proxy_in, [{'port_name': 'in', 'music_channel': c}
                          for c in range(NUM_ENC_NEURONS)])
nest.SetAcceptableLatency('in', params['delay'])  # useless?


proxy_out = nest.Create('music_rate_out_proxy')
nest.SetStatus(proxy_out, {'port_name': 'out'})

#######################################
# Create neurons

input_neurons = nest.Create(params['neuron_model'], NUM_ENC_NEURONS, input_params)
output_neurons = nest.Create(params['neuron_model'], NUM_DEC_NEURONS, output_params)
# inh_wta = nest.Create(params['neuron_model'], 1, output_params)


# critic_neurons = nest.Create('lin_rate_ipn', 1, critic_params)
critic_neurons = nest.Create('thresholdlin_rate_ipn', 1, critic_params)
# critic_neurons = nest.Create('exp_rate_ipn', 1, critic_params_exp)

reward_neurons = nest.Create('lin_rate_ipn', 1, reward_params)
# reward_pos_neurons = nest.Create(params['neuron_model'], 1, reward_thresh_params)
# reward_neg_neurons = nest.Create(params['neuron_model'], 1, reward_thresh_params)

#######################################
# Create devices

nest.SetDefaults('multimeter', {'interval': 10., 'record_from': ['rate']})
multimeter_inp = nest.Create('multimeter')
multimeter_out = nest.Create('multimeter')
multimeter_critic = nest.Create('multimeter')
multimeter_reward = nest.Create('multimeter')

wr = nest.Create('weight_recorder', 1, {
                 'label': 'weight_recorder', 'to_file': True, 'to_screen': False, 'interval': 100})

#######################################
# Create connections

nest.SetDefaults('hebbian_rate_connection', {'vt': reward_neurons[
                 0], 'n_threshold': 0., 'weight_recorder': wr[0]})

# input -> actor 
nest.Connect(input_neurons, output_neurons, 'all_to_all', {
    'model': 'hebbian_rate_connection', 'weight': 0.0,
    'A': input_actor_params['A'], 'weight_decay_constant': input_actor_params['weight_decay_constant'],
    'Wmax': input_actor_params['Wmax'], 'Wmin': input_actor_params['Wmin'], 'post_threshold': input_actor_params['post_threshold']})



# wta 
# nest.Connect(output_neurons, inh_wta, 'all_to_all', {'model': 'rate_connection', 'weight': beta_2})
# nest.Connect(inh_wta, output_neurons, 'all_to_all', {'model': 'rate_connection', 'weight': beta_1})

# for n0 in output_neurons:
#     nest.Connect([n0], [n0], 'all_to_all', {'model': 'rate_connection', 'weight': alpha})
#     for n1 in output_neurons:
#         if n0 != n1:
#             nest.Connect([n0], [n1], 'all_to_all', {'model': 'rate_connection', 'weight': -0.1 * alpha})

for n0 in output_neurons:
    for n1 in output_neurons:
        if n0 == n1:
            nest.Connect([n0], [n1], 'all_to_all', {'model': 'rate_connection', 'weight': 0.65})
        else:
            nest.Connect([n0], [n1], 'all_to_all', {'model': 'rate_connection', 'weight': -.55})

# input -> critic
nest.Connect(input_neurons, critic_neurons, 'all_to_all', {
    'model': 'hebbian_rate_connection', 'weight': 0.0,
    'A': input_critic_params['A'], 'weight_decay_constant': input_critic_params['weight_decay_constant'],
    'Wmax': input_critic_params['Wmax'], 'Wmin': input_critic_params['Wmin'], 'post_threshold': input_critic_params['post_threshold']})

# nest.SetStatus(nest.GetConnections([input_neurons[12]], critic_neurons), {"Wmax": 0.05, "Wmin": -0.05})

# critic -> reward
# nest.Connect(critic_neurons, reward_pos_neurons, 'all_to_all', {
#     'model': 'rate_connection',
#     'weight': 1. * (1. / critic_reward_params['delay'] - 1. / critic_reward_params['tau_r']),
# })
# nest.Connect(critic_neurons, reward_pos_neurons, 'all_to_all', {
#     'model': 'delay_rate_connection',
#     'weight': -1. / critic_reward_params['delay'],
#     'delay': critic_reward_params['delay']
# })
# nest.Connect(critic_neurons, reward_neg_neurons, 'all_to_all', {
#     'model': 'rate_connection',
#     'weight': -1. * (1. / critic_reward_params['delay'] - 1. / critic_reward_params['tau_r']),
# })
# nest.Connect(critic_neurons, reward_neg_neurons, 'all_to_all', {
#     'model': 'delay_rate_connection',
#     'weight': 1. / critic_reward_params['delay'],
#     'delay': critic_reward_params['delay']
# })

# nest.Connect(reward_pos_neurons, reward_neurons, 'one_to_one',
#              syn_spec={'model': 'rate_connection', 'weight': 1.})

# nest.Connect(reward_neg_neurons, reward_neurons, 'one_to_one',
#              syn_spec={'model': 'rate_connection', 'weight': -1.})


nest.Connect(critic_neurons, reward_neurons, 'all_to_all', {
   'model': 'rate_connection',
   'weight': 10. * (1. / critic_reward_params['delay'] - 1. / critic_reward_params['tau_r']),
})
nest.Connect(critic_neurons, reward_neurons, 'all_to_all', {
   'model': 'delay_rate_connection',
   'weight': -10. / critic_reward_params['delay'],
   'delay': critic_reward_params['delay']
})

c = nest.GetConnections(input_neurons,  output_neurons)
for c_ in c:
    # nest.SetStatus([c_], {'weight': 0.2 + 0.2 * np.random.rand()})
    # nest.SetStatus([c_], {'weight': -0.1 + 0.2 * np.random.rand()})
    nest.SetStatus([c_], {'weight': 0.3})

# devices -> neurons
nest.Connect(multimeter_inp, input_neurons, syn_spec={'delay': params['delay']})
nest.Connect(multimeter_out, output_neurons, syn_spec={'delay': params['delay']})
nest.Connect(multimeter_critic, critic_neurons, syn_spec={'delay': params['delay']})
nest.Connect(multimeter_reward, reward_neurons, syn_spec={'delay': params['delay']})

# MUSIC proxies -> neuron
nest.Connect(proxy_in, input_neurons, 'one_to_one',
             syn_spec={'model': 'rate_connection', 'weight': .5})


nest.Connect(reward_in, reward_neurons, 'one_to_one',
            syn_spec={'model': 'rate_connection', 'weight': 1.})

# nest.Connect(reward_in, reward_pos_neurons, 'one_to_one',
#              syn_spec={'model': 'rate_connection', 'weight': 1.0})
# nest.Connect(reward_in, reward_neg_neurons, 'one_to_one',
#              syn_spec={'model': 'rate_connection', 'weight': -1.0})

for i, v in enumerate(range(NUM_DEC_NEURONS)):
    nest.Connect([output_neurons[i]], proxy_out, 'one_to_one', {
                 'model': 'rate_connection', 'receptor_type': i})


print 'simulate'

comm.Barrier()
start = datetime.now()

nest.Simulate(to_ms(options.simtime))

end = datetime.now()
dt = end - start
run_time = dt.seconds + dt.microseconds / 1000000.

print
print
print 'RUN TIME:', run_time
print
print

'''
Analyse data and plot
'''

colormap = plt.cm.gist_ncar
# plt.gca().set_prop_cycle([colormap(i) for i in np.linspace(0, 0.9, 20)])


plt.figure('inp')
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 18)])
data = nest.GetStatus(multimeter_inp)
times = data[0]['events']['times']
senders = data[0]['events']['senders']
rates_inp = data[0]['events']['rate']
us = np.unique(senders)
plt.axhline(0.0, lw=2, ls=':', alpha=0.5, color='0.7')
for i, s in enumerate(us):
    if i % 2 == 0:
        ls = '-'
    else:
        ls= '--'
    plt.plot(times[np.where(senders == s)], rates_inp[np.where(senders == s)], label=i, ls=ls)

#rates_goal = rates_inp[np.where(senders == sorted(us)[-1])]
#t = []
#for tp in np.where(abs(rates_goal - 1.) < 1e-3)[0]:
#    if tp < len(rates_goal) - 1 and abs(rates_goal[tp + 1] - 1.) > 1e-2:
#        t.append(tp)
#plt.plot(t, np.ones_like(t), 'ro')

#plt.figure('out')
data = nest.GetStatus(multimeter_out)
times = data[0]['events']['times']
senders = data[0]['events']['senders']
rates_out = data[0]['events']['rate']
us = np.unique(senders)
plt.axhline(1.5, lw=2, ls=':', alpha=1.5, color='0.7')
for i, s in enumerate(us):
    plt.plot(times[np.where(senders == s)], rates_out[np.where(senders == s)] + 1.5, label='actor_{}'.format(i))


# plt.figure('reward')
data = nest.GetStatus(multimeter_reward)
times = data[0]['events']['times']
senders = data[0]['events']['senders']
rates_reward = data[0]['events']['rate'] * 2.
us = np.unique(senders)
plt.axhline(4.0, lw=2, ls=':', alpha=0.5, color='0.7')
for i, s in enumerate(us):
    plt.plot(times[np.where(senders == s)], rates_reward[np.where(senders == s)] + 4.0, lw=2, ls='--', c='k', label='error')

# plt.figure('value')
data = nest.GetStatus(multimeter_critic)
times = data[0]['events']['times']
senders = data[0]['events']['senders']
rates_value = data[0]['events']['rate'] * 2.
us = np.unique(senders)
plt.axhline(5.5, lw=2, ls=':', alpha=0.5, color='0.7')
for i, s in enumerate(us):
    plt.plot(times[np.where(senders == s)], rates_value[np.where(senders == s)] + 5.5, lw=2, ls='--', c='y', label='value')

plt.ylim([-1, 7])
plt.legend(loc='upper left', prop={'size': 10})

trial_length = 110
num_trials = int(to_ms(options.simtime) / 10) / int(trial_length) # multimeter has a recording interval of 10 steps

print trial_length, num_trials

plt.figure()
plt.subplot(211)
plt.gca().set_title('value')
for i in range(num_trials):
    plt.plot(rates_value[i*trial_length: (i+1)*trial_length], color=str(1. * i / num_trials))
plt.subplot(212)
plt.gca().set_title('error')
for i in range(num_trials):
    plt.plot(rates_reward[i*trial_length: (i+1)*trial_length], color=str(1. * i / num_trials))

#for i, tp in enumerate(t):
#    if tp - 60 > 0:
#        plt.plot(rates_value[tp - 60:tp], color=str(1. * i / len(t)))
#plt.subplot(212)
#plt.gca().set_title('error')
#for i, tp in enumerate(t):
#    if tp - 60 > 0:
#        plt.plot(rates_reward[tp - 60:tp], color=str(1. * i / len(t)))


# plt.figure('reward')
# data = nest.GetStatus(multimeter_reward)
# senders = data[0]['events']['senders']
# rates = data[0]['events']['rate']
# us = np.unique(senders)
# for i, s in enumerate(us):
#     plt.plot(rates[np.where(senders == s)])


def get_weights_data(wt, sources, targets):
    data = nest.GetStatus(wr)
    times = data[0]['events']['times']
    senders = data[0]['events']['senders']
    receivers = data[0]['events']['targets']
    weights = data[0]['events']['weights']

    # if sources is None:
    #     sources = np.unique(senders)
    # if targets is None:
    #     targets = np.unique(receivers)

    time_data = np.zeros(len(np.unique(times)))
    weight_data = np.zeros((len(np.unique(times)), len(sources), len(targets)))

    delta_t = np.diff(np.unique(times))[0]
    for i in xrange(len(times)):
        if senders[i] in sources and receivers[i] in targets:
            time_data[int(times[i] / delta_t) - 1] = times[i]
            weight_data[int(times[i] / delta_t) - 1][senders[i] - np.min(sources)][receivers[i] - np.min(targets)] = weights[i]

    return time_data, weight_data


def plot_weights(wr, sources, targets, label=''):
    plt.figure(label)

    times, weights = get_weights_data(wr, sources, targets)

    # plot weight course for each synapse
    plt.xlabel('time (ms)')
    plt.ylabel('weight')
    plt.rcParams.update({'font.size': 22})
    for i in sources:
        for j in targets:
            plt.plot(times, weights[:, i - np.min(sources), j - np.min(targets)], label='pre: {}({}), post: {}'.format(i, i - np.min(sources), j))
    plt.legend(loc='upper left', prop={'size': 10})


plot_weights(wr, input_neurons, output_neurons, 'actor_weights')
plot_weights(wr, input_neurons, critic_neurons, 'critic_weights')

arrow_left = np.array([-1., 0])
arrow_right = np.array([1., 0])
arrow_none = np.array([0., 1.])

times_critic, weights_critic = get_weights_data(wr, input_neurons, critic_neurons)
times_actor, weights_actor = get_weights_data(wr, input_neurons, output_neurons)
vector_actor_begin = []
vector_actor_end = []
for i in xrange(len(input_neurons)):
    vector_actor_begin.append([weights_actor[0][i][0] * arrow_left, weights_actor[0][i][1] * arrow_none, weights_actor[0][i][2] * arrow_right])
    vector_actor_end.append([weights_actor[-1][i][0] * arrow_left, weights_actor[-1][i][1] * arrow_none, weights_actor[-1][i][2] * arrow_right])

SQRT_NUM_ENC_NEURONS = int(np.sqrt(NUM_ENC_NEURONS))

plt.figure()
plt.subplot(211)
plt.pcolormesh(weights_critic[0].reshape(SQRT_NUM_ENC_NEURONS, SQRT_NUM_ENC_NEURONS), cmap='coolwarm', vmin=-.5, vmax=.5)
for i in xrange(len(input_neurons)):
    plt.arrow(0.5 + i / SQRT_NUM_ENC_NEURONS, 0.5 + i % SQRT_NUM_ENC_NEURONS, np.mean(vector_actor_begin[i], axis=0)[0], np.mean(vector_actor_begin[i], axis=0)[1], fc='k', ec='k', lw=2)
    for j in xrange(3):
        plt.arrow(0.5 + i / SQRT_NUM_ENC_NEURONS, 0.5 + i % SQRT_NUM_ENC_NEURONS, vector_actor_begin[i][j][0], vector_actor_begin[i][j][1], fc='0.7', ec='0.7', lw=1)
plt.colorbar()

plt.subplot(212)
plt.pcolormesh(weights_critic[-1].reshape(SQRT_NUM_ENC_NEURONS, SQRT_NUM_ENC_NEURONS), cmap='coolwarm', vmin=-.5, vmax=.5)
for i in xrange(len(input_neurons)):
    plt.arrow(0.5 + i / SQRT_NUM_ENC_NEURONS, 0.5 + i % SQRT_NUM_ENC_NEURONS, np.mean(vector_actor_end[i], axis=0)[0], np.mean(vector_actor_end[i], axis=0)[1], fc='k', ec='k', lw=2)
    for j in xrange(3):
        plt.arrow(0.5 + i / SQRT_NUM_ENC_NEURONS, 0.5 + i % SQRT_NUM_ENC_NEURONS, vector_actor_end[i][j][0], vector_actor_end[i][j][1], fc='0.7', ec='0.7', lw=1)
plt.colorbar()

plt.show()
