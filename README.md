# closed-loop-learning-in-autonomous-agents

## Introduction

This repository contains simulation and analysis scripts to reproduce two figures of the publication "A closed-loop toolchain for neural networksimulations of learning autonomous agents".

The network is implemented in PyNEST and can be found in ``actor_critic_network/network.py`.`

## Installation guide
The YAML file provided should be  used to set up a dedicated Python environment using (Miniconda)[https://docs.conda.io/en/latest/miniconda.html]. After installing Miniconda the environment can be created:
```bash
$ conda env create --file environment.yml
```
Additional dependencies must be installed manually:
- install MUSIC from https://github.com/INCF/MUSIC
- install NEST with MPI and MUSIC support from https://github.com/nest/nest-simulator
- install MUSIC-Adapters from https://github.com/incf-music/music-adapters

## Reproduce Fig 3, "Mountain Car"
```bash
$ cd figure_3_mountain_car
$ gymz-controller gym MountainCar-v0.json&
$ mpirun -np 6 music nest_mc.music
```
You might need to pass the `--oversubscribe` to mpirun, depending on your MPI library version.

## Reproduce Fig 4, "Frozen Lake"
