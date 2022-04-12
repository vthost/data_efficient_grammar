#!/bin/bash

set -e

REPO=$PWD
echo "REPO" $REPO

eval "$(conda shell.bash hook)"
conda activate DEG

export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_VISIBLE_DEVICES=0

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    DATAID=isocyanates
  else
    DATAID=$1
fi

DATASET="data/${DATAID}.txt"

# minimal similarity each generated molecule should have with at least one of input molecules
# (just used as filter, in order to filter out very small molecules or similar, so set as low as poossible/useful)
MINSIM=0.1

# ';' separated string of 'metric,weight' pairs
# metric can be a custom function of form 'module.functionname' which will then be imported at runtime
# see example mymetric.py
# ------------
# metrics:
# diversity:
# sharing: rules should be such that they capture aspects of more than one of the input graphs
# frags: check if required fragments are present in generated samples
# rings: check if the kinds of rings in generated samples are also found in input
# mymetric.my_metric: a custom metric doing nothing just there to show how to include external code
# syn: retro-star synthesizability score for molecule
#
# also available (better do not use for now or choose good weights or adapt implementation):
# num_rules: really just that, not normalized, ... lower considered better in order to have compact grammar
# num_samples: really just that, not normalized, inverted or similar
METRICS="diversity,1;sharing,2;frags,2;rings,2;mymetric.my_metric,2"

# ';' separated string of 'fragmentname,mincount-in-molecule', i.e., ','-separated pairs
# for valid fragment names see https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html, part after 'fr_'
FRAGS="ester,1"

# for probing
#NEPOCHS=1
#NMCMC=1
#NSAMPLES=3

NEPOCHS=50
NMCMC=5
NSAMPLES=20

python scripts/deg_main.py --training_data=$DATASET  --min_sim=$MINSIM --metrics=$METRICS --req_frags=$FRAGS \
        --max_epochs=$NEPOCHS --num_generated_samples=$NSAMPLES  --MCMC_size=$NMCMC  &> "out_${DATAID}.txt"
