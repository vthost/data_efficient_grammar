import logging
import torch
import os
import pprint
import pickle
import argparse
import importlib
import numpy as np
import torch.optim as optim
from copy import deepcopy
from inspect import isfunction

from private import *
from private.metrics2 import *
from deg_gen import check_generated_mol, random_produce
from main import retro_sender
from agent import Agent
from grammar_generation import data_processing, MCMC_sampling

# num_rules: lower considered better in order to have compact grammar
# num_samples: really just that, not normalized, inverted or similar - better do not use for now or adapt implementation
DEFAULT_METRICS = ['diversity', 'num_rules', 'num_samples', 'syn', 'sharing', 'rings', 'frags']


def evaluate(grammar, input_graphs, args, metrics, custom_metrics={}):
    print("Start grammar evaluation...")

    if not hasattr(args, 'inp_smiles'):  # compute once only
        args.inp_smiles, args.inp_mols = [g.smiles for g in input_graphs], [g.mol for g in input_graphs]
        args.inp_fps, args.inp_rings = get_fps(args.inp_mols), get_ring_info(args.inp_mols)

    idx, no_newly_generated_iter = 0, 0
    generated_mols, generated_smiles = [], []
    while idx < args.num_generated_samples and no_newly_generated_iter <= 10:
        print("Generating sample {}/{}".format(idx, args.num_generated_samples))
        mol, iter_num, (rules, rules_idx) = random_produce(grammar)
        if mol is None:
            no_newly_generated_iter += 1
            continue
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))

        if check_generated_mol(smiles, mol, rules_idx, args.inp_smiles, args.inp_fps, args.min_sim, gen_others_smi=generated_smiles):
            generated_mols.append(mol)
            generated_smiles.append(smiles)
            idx += 1
            no_newly_generated_iter = 0
        else:
            no_newly_generated_iter += 1

    # Metric evalution for the given grammar
    eval_metrics = {}
    div = InternalDiversity()
    num_samples = len(generated_mols)
    for _metric in metrics:
        assert _metric in DEFAULT_METRICS or _metric in custom_metrics
        if _metric == 'diversity':
            diversity = div.get_diversity(generated_mols)
            eval_metrics[_metric] = diversity
        elif _metric == 'num_rules':
            eval_metrics[_metric] = grammar.num_prod_rule
        elif _metric == 'num_samples':
            eval_metrics[_metric] = idx
        elif _metric == 'syn':
            eval_metrics[_metric] = retro_sender(generated_mols, args)
        elif _metric == 'sharing':
            eval_metrics[_metric] = rule_sharing(grammar, input_graphs)
        elif _metric == 'frags':
            eval_metrics[_metric] = 0
            for frag, ct in args.req_frags:
                eval_metrics[_metric] += sum(list(map(lambda mol: int(frag_presence(mol, frag, min=ct)),
                                                     generated_mols)))
            eval_metrics[_metric] = eval_metrics[_metric] / (num_samples * len(args.req_frags))
        elif _metric == 'rings':
            eval_metrics[_metric] = sum(list(map(lambda mol: int(rings_in_input(mol, args.inp_rings)),
                                                 generated_mols))) / num_samples
        else:
            func = custom_metrics[_metric]
            eval_metrics[_metric] = sum(list(map(lambda t: func(t[0], t[1]), zip(generated_mols, generated_smiles)))) / num_samples
            print("custom is", eval_metrics[_metric])
    return eval_metrics


def combine_metrics(result, metrics):
    r, ct = 0, 0
    for metric, weight in metrics:
        r += weight*result[metric]
        ct += weight
    return r / ct


def learn(smiles_list, args):
    metric_names = [m[0] for m in args.metrics]
    # Create logger
    dataid = args.training_data[args.training_data.rindex("/")+1:args.training_data.rindex(".")]
    save_log_path = f'./data/grammar/{dataid}/'

    create_exp_dir(save_log_path) #, scripts_to_save=[f for f in os.listdir('./') if f.endswith('.py')])
    logger = create_logger('global_logger', save_log_path + '/log.txt')
    logger.info('args:{}'.format(pprint.pformat(args)))
    logger = logging.getLogger('global_logger')
    args.logger = logger

    # Initialize dataset & potential function (agent) & optimizer
    subgraph_set_init, input_graphs_dict_init = data_processing(smiles_list, args.GNN_model_path, args.motif)
    agent = Agent(feat_dim=300, hidden_size=args.hidden_size)
    if args.resume:
        assert os.path.exists(args.resume_path), "Please provide valid path for resuming."
        ckpt = torch.load(args.resume_path)
        agent.load_state_dict(ckpt)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # Start training
    logger.info('starting\n')
    curr_max_R = 0
    for train_epoch in range(args.max_epochs):
        returns = []
        log_returns = []
        logger.info("<<<<< Epoch {}/{} >>>>>>".format(train_epoch, args.max_epochs))

        # MCMC sampling
        for num in range(args.MCMC_size):
            grammar_init = ProductionRuleCorpus()
            l_input_graphs_dict = deepcopy(input_graphs_dict_init)
            l_subgraph_set = deepcopy(subgraph_set_init)
            l_grammar = deepcopy(grammar_init)
            iter_num, l_grammar, l_input_graphs_dict = MCMC_sampling(agent, l_input_graphs_dict, l_subgraph_set, l_grammar, num, args)
            # Grammar evaluation
            eval_metric = evaluate(l_grammar, list(l_input_graphs_dict.values()), args, metric_names, custom_metrics=args.custom_metrics)
            logger.info("eval_metrics: {}".format(eval_metric))
            # Record metrics
            R = combine_metrics(eval_metric, args.metrics)
            R_ind = R  #.copy()
            returns.append(R)
            log_returns.append(eval_metric)
            logger.info("====== MCMC {} sampling score {}=======:".format(num, R_ind))
            # Save ckpt
            if R_ind > curr_max_R:
                # torch.save(agent.state_dict(), os.path.join(save_log_path, 'agent_{}_{}.pkl'.format(train_epoch, num)))
                torch.save(l_grammar, '{}/grammar_{}_{}.pt'.format(save_log_path, train_epoch, num), pickle_protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(l_input_graphs_dict, '{}/input_graphs_orig_{}_{}.pt'.format(save_log_path, train_epoch, num), pickle_protocol=pickle.HIGHEST_PROTOCOL)
                mols = [g.mol for g in l_input_graphs_dict.values()]
                smis = [g.smiles for g in l_input_graphs_dict.values()]
                ridx = [g.rule_idx_list for g in l_input_graphs_dict.values()]
                torch.save((mols, smis, ridx),
                           '{}/input_graphs_{}_{}.pt'.format(save_log_path, train_epoch, num),
                           pickle_protocol=pickle.HIGHEST_PROTOCOL)

                curr_max_R = R_ind
        # VT didn't change below
        # Calculate loss
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) # / (returns.std() + eps)
        assert len(returns) == len(list(agent.saved_log_probs.keys()))
        policy_loss = torch.tensor([0.])
        for sample_number in agent.saved_log_probs.keys():
            max_iter_num = max(list(agent.saved_log_probs[sample_number].keys()))
            for iter_num_key in agent.saved_log_probs[sample_number].keys():
                log_probs = agent.saved_log_probs[sample_number][iter_num_key]
                for log_prob in log_probs:
                    policy_loss += (-log_prob * args.gammar ** (max_iter_num - iter_num_key) * returns[sample_number]).sum()

        # Back Propogation and update
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        agent.saved_log_probs.clear()

        # Log
        logger.info("Loss: {}".format(policy_loss.clone().item()))
        eval_metrics = {}
        for r in log_returns:
            for _key in r.keys():
                if _key not in eval_metrics:
                    eval_metrics[_key] = []
                eval_metrics[_key].append(r[_key])
        mean_evaluation_metrics = ["{}: {}".format(_key, np.mean(eval_metrics[_key])) for _key in eval_metrics]
        logger.info("Mean evaluation metrics: {}".format(', '.join(mean_evaluation_metrics)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCMC training')
    parser.add_argument('--training_data', type=str, default="./data/isocyanates.txt")
    parser.add_argument('--GNN_model_path', type=str, default="./GCN/model_gin/supervised_contextpred.pth", help="file name of the pretrained GNN model")
    parser.add_argument('--sender_file', type=str, default="generated_samples.txt", help="file name of the generated samples")
    parser.add_argument('--receiver_file', type=str, default="output_syn.txt", help="file name of the output file of Retro*")
    parser.add_argument('--resume', action="store_true", default=False, help="resume model")
    parser.add_argument('--resume_path', type=str, default='', help="resume path")
    parser.add_argument('--learning_rate', type=int, default=1e-2, help="learning rate")
    parser.add_argument('--gammar', type=float, default=0.99, help="discount factor")
    parser.add_argument('--motif', action="store_true", default=False, help="use motif as the basic building block for polymer dataset")
    parser.add_argument('--hidden_size', type=int, default=128, help="hidden size of the potential function")
    parser.add_argument('--max_epochs', type=int, default=50, help="maximal training epoches")  # 50
    parser.add_argument('--MCMC_size', type=int, default=5, help="sample number of each step of MCMC")
    parser.add_argument('--num_generated_samples', type=int, default=20, help="number of generated samples to evaluate grammar")
    parser.add_argument('--min_sim', type=float, default=0.1, help="min similarity of generated samples to at least one sample in input")
    parser.add_argument('--req_frags', type=str, default="",
                        help="semic colon, comma separated string of functional groups & min counts to be contained in generated molecules, "
                             "for names see https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html, part after 'fr_', "
                             "e.g., 'ester,1;Ar_COO,2' ")
    # one of ['diversity', 'num_rules', 'num_samples', 'syn', 'sharing', 'rings', 'frags'] OR custom
    # custom currently must be in script directory, not necessarily imported yet
    parser.add_argument('--metrics', type=str, default="diversity,1;sharing,2;frags,2;rings,2;mymetric.my_metric,2")
    args = parser.parse_args()
    np.random.seed(0)

    assert os.path.exists(args.training_data), "Please provide valid path of training data."
    # Remove duplicated molecules
    with open(args.training_data, 'r') as fr:
        lines = fr.readlines()
        mol_sml = []
        for line in lines:
            if not (line.strip() in mol_sml):
                mol_sml.append(line.strip())
            else:
                print("duplicate")

    # for probing
    # args.max_epochs = 1
    # args.MCMC_size = 1
    # args.num_generated_samples = 3
    # args.req_frags = "ester,1"
    # mol_sml = mol_sml[:2]

    # Clear the communication files for Retro*
    with open(args.sender_file, 'w') as fw:
        fw.write('')
    with open(args.receiver_file, 'w') as fw:
        fw.write('')

    def prep_arg(arg):
        arg = [tuple(s.split(',')) for s in arg.split(';')]
        return [(s[0], int(s[1])) for s in arg]

    args.req_frags = prep_arg(args.req_frags)
    args.metrics = prep_arg(args.metrics)
    args.custom_metrics = {}
    for m, _ in args.metrics:
        if m not in DEFAULT_METRICS:
            module = m[:m.rindex(".")]
            module = importlib.import_module(module)
            func = getattr(module, m[m.rindex(".")+1:])
            assert isfunction(func), f"Invalid custom metric, {m} is no function"
            args.custom_metrics[m] = func

    # Grammar learning
    learn(mol_sml, args)

    # TODO check if this should be kept, then move into save directory
    os.remove(args.sender_file)
    os.remove(args.receiver_file)

