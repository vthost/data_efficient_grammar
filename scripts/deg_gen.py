import os
import pickle5 as pickle
import argparse
import numpy as np
from rdkit import Chem
from copy import deepcopy
import torch
from private import *  # needed for loading .pt files
from private.metrics2 import *


# VT this is mostly same as in grammar_generation but added (rules, rules_idx) to return
def random_produce(grammar):
    def sample(l, prob=None):
        if prob is None:
            prob = [1/len(l)] * len(l)
        idx = np.random.choice(range(len(l)), 1, p=prob)[0]
        return l[idx], idx

    def prob_schedule(_iter, selected_idx):
        prob_list = []
        # prob = exp(a * t * x), x = {0, 1}
        a = 0.5
        for rule_i, rule in enumerate(grammar.prod_rule_list):
            x = rule.is_ending
            if rule.is_start_rule:
                prob_list.append(0)
            else:
                prob_list.append(np.exp(a * _iter * x))
        prob_list = np.array(prob_list)[selected_idx]
        prob_list = prob_list / np.sum(prob_list)
        return prob_list

    hypergraph = Hypergraph()
    rules, rules_idx = [], []
    starting_rules = [(rule_i, rule) for rule_i, rule in enumerate(grammar.prod_rule_list) if rule.is_start_rule]
    iter = 0
    while(True):
        if iter == 0:
            _, idx = sample(starting_rules)
            selected_rule_idx, selected_rule = starting_rules[idx]
            hg_cand, _, avail = selected_rule.graph_rule_applied_to(hypergraph)
            hypergraph = deepcopy(hg_cand)
            rules += [selected_rule]
            rules_idx += [selected_rule_idx]
        else:
            candidate_rule = []
            candidate_rule_idx = []
            candidate_hg = []
            for rule_i, rule in enumerate(grammar.prod_rule_list):
                hg_cand, _, avail = rule.graph_rule_applied_to(hypergraph)
                if(avail):
                    candidate_rule.append(rule)
                    candidate_rule_idx.append(rule_i)
                    candidate_hg.append(hg_cand)
            if (all([rl.is_start_rule for rl in candidate_rule]) and iter > 0) or iter > 30:
                break
            prob_list = prob_schedule(iter, candidate_rule_idx)
            hypergraph, idx = sample(candidate_hg, prob_list)

            rules += [candidate_rule[idx]]
            rules_idx += [candidate_rule_idx[idx]]
        iter += 1
    try:
        mol = hg_to_mol(hypergraph)
        Chem.SanitizeMol(mol)  # try here directly if this works
        # print(Chem.MolToSmiles(mol))

    except Exception as e:
        print(e)
        return None, iter, ([], [])

    return mol, iter, (rules, rules_idx)


# gen_others_smi or gen_others_rule_idx may be empty, depending on which is relevant in current context
# eg for training we may want same smiles with different rule indices so gen_others_smi=[]
def check_generated_mol(smiles, mol, rule_idx, inp_smiles, inp_fps, min_sim, gen_others_smi=[], gen_others_rule_idx=[]):
    # exclude if smi in inp_smi eg to have no duplicates from test in train
    if smiles in inp_smiles \
            or smiles in gen_others_smi \
            or rule_idx in gen_others_rule_idx \
            or not min_inp_similarity(mol, inp_fps, min_sim=min_sim):
        return False

    return True


def check_generated_mol_extended(smiles, mol, rule_idx, inp_smiles, inp_fps, min_sim, inp_rings, req_frags=[],
                                 gen_others_smi=[], gen_others_rule_idx=[]):

    if not check_generated_mol(smiles, mol, rule_idx, inp_smiles, inp_fps, min_sim, gen_others_smi=gen_others_smi,
                               gen_others_rule_idx=gen_others_rule_idx):
        return False

    for frag, ct in req_frags:
        if not frag_presence(mol, frag, ct):
            return False

    if not rings_in_input(mol, inp_rings):
        return False

    return True


def random_produce_graphs(grammar_file, save_file, n=100, input_graphs=False, req_frags=[]):
    grammar = torch.load(grammar_file)
    mols, mols_rule_idx, mols_smi = [], [], []
    inp_smi, inp_rule_idx, inp_fps, inp_rings = [], [], [], []

    if input_graphs:
        inp_mols, inp_smi, inp_rule_idx = \
            torch.load(grammar_file.replace("grammar_", "input_graphs_"))

        inp_fps, inp_rings = get_fps(inp_mols), get_ring_info(inp_mols)

    for i in range(n):
        success = False
        while not success:
            mol, _, (rules, rules_idx) = random_produce(grammar)
            if mol is None:
                continue

            smi = Chem.MolToSmiles(mol)
            if not check_generated_mol_extended(smi, mol, rules_idx, inp_smi, inp_fps, args.min_sim, inp_rings,
                                                req_frags=req_frags, gen_others_rule_idx=mols_rule_idx):
                continue

            mols += [mol]
            mols_smi += [smi]
            mols_rule_idx += [rules_idx]
            success = True

    if save_file is not None:
        os.makedirs(save_file[:save_file.rindex("/")], exist_ok=True)
        # collect labels (rdkit doesn't save the int props somehow, saving extra is easier than adapting serialization)
        # do this before extension of mols!!

        if input_graphs:
            mols_smi += inp_smi

        with open(save_file + "_rindex.txt", "w") as f:
            for i, smi in enumerate(mols_smi):
                idxs = ";".join([str(j) for j in mols_rule_idx[i]])
                f.write(f"{smi},{idxs}\n")
        with open(save_file + ".txt", "w") as f:
            f.write("\n".join(mols_smi))

    return mols, mols_smi, mols_rule_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEG generation')
    parser.add_argument('--num_samples', type=int, default=100,
                        help="number of generated samples")
    parser.add_argument('--grammar', type=str, default=None, help="file")
    parser.add_argument('--save', type=str, default=None, help="save path")
    parser.add_argument('--min_sim', type=float, default=0.1, help="min similarity of generated samples to at least one in input")
    parser.add_argument('--req_frags', type=str, default="",
                        help="semic colon, comma separated string of functional groups & min counts to be contained in generated molecules, "
                             "for names see https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html, part after 'fr_', "
                             "e.g., 'ester,1;Ar_COO,2' ")
    args = parser.parse_args()
    np.random.seed(0)

    # for probing
    # args.grammar = "./data/grammar/isocyanates/grammar_0_0.pt"
    # args.num_samples = 3
    # args.req_frags = "ester,1"

    assert os.path.exists(args.grammar), "Please provide valid path for grammar."

    if args.save is None:
        args.save = args.grammar[:args.grammar.rindex("/")].replace("grammar", "generated")

    random_produce_graphs(args.grammar, args.save, args.num_samples, input_graphs=1,
                          req_frags=[tuple(s.split(',')) for s in args.req_frags.split(';')])

