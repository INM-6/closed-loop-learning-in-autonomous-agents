# closed-loop-learning-in-autonomous-agents

This repository provides scripts to reproduce the figures of the publication "A closed-loop toolchain for neural networksimulations of learning autonomous agents".


## installation guide
- conda env create --file environment.yml
- install MUSIC from https://github.com/INCF/MUSIC
- install NEST with MPI and MUSIC support from https://github.com/nest/nest-simulator
- install MUSIC-Adapters from https://github.com/incf-music/music-adapters

## run simulations for Fig 3 "Mountain Car"



gymz-controller gym MountainCar-v0.json
