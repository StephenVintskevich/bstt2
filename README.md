# bstt

### Instalation and Usage:

Using Anaconda:

conda create --name bstt

conda activate bstt

conda install numpy matplotlib scipy scikit-learn python3

See the experiments for example usage.


### File Structure

First level, /:

bstt.py: block sparse tensor trains

als.py: als algorithms

misc.py: helpers for bstt.py and als.py

helpers.py: this is the code for the dynamic models, e.g. fermi pasta, lennard jones etc.

Second level: expermiments/

The folder for each system contains a file to generate the data for each example dynamical system, a plotting file and folders containing data and figures. Note that each script is to be run from within the folder corresponding to the dynamical system at hand.

1. FPUT

2. Magnetic dipole chain:

exp_1_magnetic.py trigonometric dictionary

exp_2_magnetic.py polynomial dictionary

exp_3_noise_magnetic.py trigonometric dictionary + gaussian noise


3. Lennard-Jones
