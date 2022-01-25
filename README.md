# bstt

First level, /:

bstt.py: code for block sparse tensor trains

als.py: code for the als algorithms

misc.py: these are helpers for bstt.py and als.py

helpers.py: this is the code for the dynamic models, e.g. fermi pasta, lennard jones etc.

Second level: expermiments_dynamical_systems/

1. magnetic dipole:

exp_1_magnetic.py trigonometric basisfunctions

exp_2_magnetic.py polynomial basisfunctions

exp_3_noise_magnetic.py trigonometric basisfunctions + gaussian noise


2. lennard jones

exp_3_lennardjones.py exp=1 and no sign and abs values and multiplying by (x_k-x_k-1)^(2exp+1), calling lennardJonesParam3Mod

exp_5_lennardjones.pyexp=1 and with sign and abs values and multiplying by (x_k-x_k-1)^(2exp+1), calling lennardJonesParam2Mod
