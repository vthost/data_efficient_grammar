import pickle
import pandas as pd
import logging
from mlp_retrosyn.mlp_inference import MLPModel
from retro_star.alg import molstar

def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols

def prepare_mlp(templates, model_dump):
    logging.info('Templates: %s' % templates)
    logging.info('Loading trained mlp model from %s' % model_dump)
    one_step = MLPModel(model_dump, templates, device=-1)
    return one_step

def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk,
        iterations, viz=False, viz_dir=None):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0: molstar(
            target_mol=x,
            target_mol_id=y,
            starting_mols=starting_mols,
            expand_fn=expansion_handle,
            value_fn=value_fn,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
            )
    return plan_handle

class prepare_molstar_planner_fn():
    def __init__(self, one_step, value_fn, starting_mols, expansion_topk, iterations, viz=False, viz_dir=None):
        self.one_step = one_step
        self.value_fn = value_fn
        self.starting_mols = starting_mols
        self.expansion_topk = expansion_topk
        self.iterations = iterations
        self.viz = viz
        self.viz_dir = viz_dir

    def expansion_handle(self, x):
        return self.one_step.run(x, topk=self.expansion_topk)

    def __call__(self, x):
        plan_handle = molstar(
            target_mol=x,
            target_mol_id=0,
            starting_mols=self.starting_mols,
            expand_fn=self.expansion_handle,
            value_fn=self.value_fn,
            iterations=self.iterations,
            viz=self.viz,
            viz_dir=self.viz_dir
        )
        return plan_handle
