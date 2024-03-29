# PM analysis and prediction task

My analysis of pm2.5 data in Beijng, years 2010-2014.
Dataset PRSA_data_2010.1.1-2014.12.31.csv downloaded from https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data#


## Prerequisites

* python3
* python3 libs listed in requirements.txt
* keras for neural network model (and backend for keras)


## Installing
pip3 install -r requirements.txt (better use python virtualenv)

In order to install keras, follow keras installation instruction.

Tested on tenserflow backend.

## Running
python3 analyze.py # to see summaries and lots of plots on dataset

python3 simple_models.py # to see linear model (with summaries) and stochastic gradient descent model

python3 nn_model.py # to see neural network model

## Presentation
To see presentation, open presentation.pdf


![picture](https://github.com/kubapok/pm-task/blob/master/presentation/time-period.png)
