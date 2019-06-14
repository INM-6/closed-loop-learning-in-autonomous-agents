#!/usr/bin/env python

from datetime import datetime
import json
import nest
import numpy as np
from optparse import OptionParser
import pylab as plt
import sys

from mpi4py import MPI  # needs to be imported after NEST

comm = MPI.COMM_WORLD


def to_ms(t):
    return t * 1000.

#######################################
# Load parameters

params_fn = 'network_params.json'

try:
    with open(params_fn, 'r') as f:
        params = json.load(f)
except TypeError:
    raise ValueError('Please provide a network parameter file.')

#######################################
# Initialize Kernel

np.random.seed(123)

nest.ResetKernel()
nest.set_verbosity('M_ERROR')
nest.SetKernelStatus({'resolution': params['kernel_params']['dt'], 'print_time': False,
                      'use_wfr': False, 'overwrite_files': True, 'grng_seed': params['kernel_params']['seed'],
                      'rng_seeds': [params['kernel_params']['seed'] + 1]})

#######################################
# Create MUSIC devices

reward_in = nest.Create('music_rate_in_proxy', 1)
nest.SetStatus(reward_in, [{'port_name': 'reward_in',
                            'music_channel': c} for c in range(1)])
nest.SetAcceptableLatency('reward_in', params['kernel_params']['delay'])  # useless?

proxy_in = nest.Create('music_rate_in_proxy', params['input_params']['num_neurons'])
nest.SetStatus(proxy_in, [{'port_name': 'in', 'music_channel': c}
                          for c in range(params['input_params']['num_neurons'])])
nest.SetAcceptableLatency('in', params['kernel_params']['delay'])  # useless?


proxy_actor = nest.Create('music_rate_out_proxy')
nest.SetStatus(proxy_actor, {'port_name': 'out'})

#######################################
# Create neurons

# extract num neurons, otherwise NEST complains about a dict entry that it does
# not understand during nest.Create
num_input_neurons = params['input_params']['num_neurons']
del params['input_params']['num_neurons']
num_actor_neurons = params['actor_params']['num_neurons']
del params['actor_params']['num_neurons']

input_neurons = nest.Create(params['kernel_params']['neuron_model'], num_input_neurons, params['input_params'])
actor_neurons = nest.Create(params['kernel_params']['neuron_model'], num_actor_neurons, params['actor_params'])
critic_neurons = nest.Create(params['kernel_params']['neuron_model'], 1, params['critic_params'])
reward_neurons = nest.Create('rate_volume_transmitter_ipn', 1, params['reward_params'])

#######################################
# Create devices

nest.SetDefaults('multimeter', {'interval': 10., 'record_from': ['rate']})
multimeter_inp = nest.Create('multimeter', 1, {
    'label': params['all']['label_prefix'] + 'in', 'to_file': True})
multimeter_actor = nest.Create('multimeter', 1, {
    'label': params['all']['label_prefix'] + 'actor', 'to_file': True})
multimeter_critic = nest.Create('multimeter', 1, {
    'label': params['all']['label_prefix'] + 'critic', 'to_file': True})
multimeter_reward = nest.Create('multimeter', 1, {
    'label': params['all']['label_prefix'] + 'reward', 'to_file': True})

wr = nest.Create('weight_recorder', 1, {
    'label': params['all']['label_prefix'] + 'weight_recorder',
    'to_file': True, 'to_screen': False, 'interval': 100})

#######################################
# Create connections

nest.SetDefaults('hebbian_rate_connection', {'vt': reward_neurons[
    0], 'n_threshold': 0., 'weight_recorder': wr[0]})

# input -> critic
nest.Connect(input_neurons, critic_neurons, 'all_to_all', params['input_critic_params'])

# critic -> reward
nest.Connect(critic_neurons, reward_neurons, 'all_to_all', {
    'model': 'rate_connection_instantaneous',
    'weight': params['critic_reward_params']['weight_scaling'] * (
        1. / params['critic_reward_params']['delay'] - 1. / params['critic_reward_params']['tau_r']),
})
nest.Connect(critic_neurons, reward_neurons, 'all_to_all', {
    'model': 'rate_connection_delayed',
    'weight': -1. * params['critic_reward_params']['weight_scaling'] / params['critic_reward_params']['delay'],
    'delay': params['critic_reward_params']['delay']
})

# input -> actor
nest.Connect(input_neurons, actor_neurons, 'all_to_all', params['input_actor_params'])

# actor wta circuit
nest.Connect(actor_neurons, actor_neurons, 'all_to_all', {'model': 'rate_connection_instantaneous', 'weight': 1.})

# devices -> neurons
nest.Connect(multimeter_inp, input_neurons, syn_spec={'delay': params['kernel_params']['delay']})
nest.Connect(multimeter_actor, actor_neurons, syn_spec={'delay': params['kernel_params']['delay']})
nest.Connect(multimeter_critic, critic_neurons, syn_spec={'delay': params['kernel_params']['delay']})
nest.Connect(multimeter_reward, reward_neurons, syn_spec={'delay': params['kernel_params']['delay']})

# MUSIC proxies -> neurons
nest.Connect(proxy_in, input_neurons, 'one_to_one',
             syn_spec={'model': 'rate_connection_instantaneous', 'weight': 0.5})

nest.Connect(reward_in, reward_neurons, 'one_to_one',
             syn_spec={'model': 'rate_connection_instantaneous', 'weight': params['reward_input_params']['weight']})

for i, v in enumerate(range(num_actor_neurons)):
    nest.Connect([actor_neurons[i]], proxy_actor, 'one_to_one', {
                 'model': 'rate_connection_instantaneous', 'receptor_type': i})

# set weights in WTA
for i, n0 in enumerate(actor_neurons):
    for j, n1 in enumerate(actor_neurons):
        dist = abs(i - j)
        if dist > num_actor_neurons / 2:
            dist = abs(dist - num_actor_neurons)

        weight = params['actor_wta_params']['exc'] * np.exp(- np.dot(dist, dist) / np.power(params['actor_wta_params']['sigma'], 2) ) + params['actor_wta_params']['inh']

        c = nest.GetConnections([n0], [n1])
        nest.SetStatus(c, {'weight': weight})


#######################################
# Simulate network

print('simulate')

comm.Barrier()
start = datetime.now()

nest.Simulate(to_ms(params['kernel_params']['simtime']))

end = datetime.now()
dt = end - start
run_time = dt.seconds + dt.microseconds / 1000000.

print()
print()
print('RUN TIME:', run_time)
print()
print()
