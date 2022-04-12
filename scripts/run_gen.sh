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
    echo "No arguments supplied, using probably the wrong grammar now"
    DATAID=isocyanates
    EPOCHID=grammar_0_0.pt
  else
    DATAID=$1
    EPOCHID=$2
fi

DATASET="${REPO}/data/grammar/${DATAID}/${EPOCHID}"

MINSIM=0.1
FRAGS="ester,1"
NSAMPLES=10

python scripts/deg_gen.py --grammar=$DATASET --req_frags=$FRAGS --min_sim=$MINSIM --num_samples=$NSAMPLES  &> "out_gen_${DATAID}.txt"
