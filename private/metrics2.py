import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import Fragments, RDKFingerprint, MACCSkeys, AllChem, Lipinski


def get_fps(mols, fp_type='ecfp4'):
    return [mol_to_fp(m, fp_type) for m in mols]


RING_OP_NAMES = ["NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
                 "NumAromaticCarbocycles", "NumAromaticHeterocycles",
                 "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
                 "RingCount"]


def get_rdkit_ring_info(mols):
    inp_rings = {op: 0 for op in RING_OP_NAMES}  # we only record 0/1 as num > 0 (False/True)
    for opname in RING_OP_NAMES:
        op = getattr(Lipinski, opname)
        for m in mols:
            if op(m) > 0:
                inp_rings[opname] = 1
                break
    return inp_rings


class Ring:    # TODO discuss what else to capture beyond atom symbols and bond types
    def __init__(self, mol, atom_idx):
        atoms = []
        for a in mol.GetAtoms():
            i = a.GetIdx()
            if i in atom_idx:
                atoms += [(a.GetSymbol(), i)]
        atoms = list(sorted(atoms))
        i2new = {i: new_i for new_i, (_, i) in enumerate(atoms)}
        atoms = [s for s, _ in atoms]

        bonds = []
        bond_types = []
        for b in mol.GetBonds():
            i1, i2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i1 not in atom_idx or i2 not in atom_idx:
                continue
            # if kekulize: TODO check
            #     is_aromatic = False
            #     if bond.GetBondType().real == 12:
            #         bond_type = 1
            #     else:
            #         bond_type = bond.GetBondType().real
            # else:
            is_aromatic = b.GetIsAromatic()
            bond_type = b.GetBondType().real
            # TODO do we need other info for comparison? then add here
            if i2new[i1] < i2new[i2]:
                bonds += [(i2new[i1], i2new[i2], atoms[i2new[i1]], atoms[i2new[i2]], bond_type, is_aromatic)]
            else:
                bonds += [(i2new[i2], i2new[i1], atoms[i2new[i2]], atoms[i2new[i1]], bond_type, is_aromatic)]
            bond_types += [(bond_type, is_aromatic)]

        self.atoms = atoms
        self.bonds = list(sorted(bonds))
        self.bond_types = set(bond_types)
        # print(atoms)
        # print(self.bonds)
        # print(self.bond_types)

    def __eq__(self, other):
        if not isinstance(other, Ring):
            return False

        # simple checks
        if len(self.atoms) != len(other.atoms):
            return False
        for a in self.atoms:  # general symbol occurrence
            if a not in other.atoms:
                return False
        for b in self.bond_types:  # general type occurrence
            if b not in other.bond_types:
                return False

        #  for each start of indexing in second ring, check eq to self
        na = len(self.atoms)
        for offset in range(na):
            bonds = [((i1+offset)%na, (i2+offset)%na, s1, s2, bt, ba) for i1, i2, s1, s2, bt, ba in other.bonds]
            bonds = list(sorted(bonds))
            # print(bonds)
            if bonds == self.bonds:
                return True

        return False


def get_rings(mol):
    rings = []
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    for atom_idx in ssr:
        r = Ring(mol, atom_idx)
        if r not in rings:
            rings += [r]
    return rings


def get_ring_info(mols):
    rdkit_ring_info = get_rdkit_ring_info(mols)  # for fast comparison
    rings = []
    for mol in mols:
        for r in get_rings(mol):
            if r not in rings:
                rings += [r]
    return rdkit_ring_info, rings


# this is a relative score depending on the number of input graphs, so not directly comparable
def rule_sharing(grammar, inp_graphs):
    cts = [0]*grammar.num_prod_rule
    for ig in inp_graphs:
        for i in ig.rule_idx_list:
            cts[i] += 1
    num_sharing = sum([ct for ct in cts if ct > 1])  # TODO how to reward large more 1 1 1 1 = 2 2 0 0 - but we also might not want too much?
    return num_sharing / grammar.num_prod_rule


# https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
def frag_presence(mol, frag_name, min=1):
    try:
        op = getattr(Fragments, f'fr_{frag_name}')
    except Exception as e:
        print(e)
        raise ValueError(f"Fragment name {frag_name} invalid.")

    result = op(mol)
    if result < int(min):  # accept strings as well
        return False
    return True


def mol_to_fp(mol, fp_type='ecfp4'):
    if fp_type == "rdk":
        return RDKFingerprint(mol)
    elif fp_type == "maacs":
        return MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "ecfp4":
        return AllChem.GetMorganFingerprint(mol, 2)
    else:
        ValueError(f"Finger print type {fp_type} invalid.")


# TODO find sim value
# ensure >= min_sim similarity to >= 1 input graph
# default fp_type was chosen arbitrarily
def min_inp_similarity(mol, inp_fps, min_sim=0.1, fp_type='ecfp4'):
    fp = mol_to_fp(mol, fp_type)
    for inp_fp in inp_fps:
        sim = DataStructs.DiceSimilarity(fp, inp_fp)  # DataStructs.FingerprintSimilarity TODO check, in the tutorial this is used for other fp types, is this necessary?
        if sim > min_sim:
            return True
    return False


def rings_in_input(mol, inp_rings):
    inp_rdkitinfo, inp_rings = inp_rings

    for opname in RING_OP_NAMES:
        op = getattr(Lipinski, opname)
        if op(mol) > 0 and not inp_rdkitinfo[opname]:
            return False

    for ring in get_rings(mol):
        if ring not in inp_rings:
            return False

    return True


if __name__ == "__main__":
    smi0 = "CCOC(=O)C"
    mol0 = Chem.MolFromSmiles(smi0)
    succ = frag_presence(mol0, "ester", 1)
    assert succ

    inp_mols = [Chem.MolFromSmiles("CC(C)(C)c1cc(-c2n[nH]c(=S)o2)cc(C(C)(C)C)c1O")]
    inp_fps = get_fps(inp_mols)

    succ0 = min_inp_similarity(mol0, inp_fps)
    assert not succ0

    mol1 = Chem.MolFromSmiles("CS(=O)(=O)Nc1ccc([N+](=O)[O-])cc1OC1CCCCC1")
    succ1 = min_inp_similarity(mol1, inp_fps)
    assert succ1

    same_rings = [
        ("c12c(cccc1)cccc2", "c1cc2ccccc2cc1"),  # naphtalene
        ("C1CCCCC1C2CCCCC2", "C0CCCCC0C0CCCCC0"),  # bicyclohexyl
        ("C1CCCCC1C2CCCCC2", "C1CCCCC1")  # bicyclohexyl, cyclohexane
    ]
    for smi2, smi3 in same_rings:
        mol2 = Chem.MolFromSmiles(smi2)
        rings2 = get_rings(mol2)  # now correctly recognizes two similar rings and returns only one
        print()
        mol3 = Chem.MolFromSmiles(smi3)
        rings3 = get_rings(mol3)
        for i, r2 in enumerate(rings2):
            for j, r3 in enumerate(rings3):
                # print(i,j)
                assert r2 == r3

    not_same_rings = [
        ("C1CC1", "C1CCCCC1C2CCCCC2"),  # cyclopropane, bicyclohexyl - diff. ring size (carbocycles)
        ("c12c(cccc1)cccc2", "C1CCCCC1C2CCCCC2"),  # naphtalene, bicyclohexyl - aromatic vs non aromatic carbocycle
        ("C1CCCCC1C2CCCCC2", "C1CCNCC1")  # bicyclohexyl vs piperidine - homo vs heterocycle
    ]

    for smi2, smi3 in not_same_rings:
        mol2 = Chem.MolFromSmiles(smi2)
        rings2 = get_rings(mol2)  # now correctly recognizes two similar rings and returns only one
        print()
        mol3 = Chem.MolFromSmiles(smi3)
        rings3 = get_rings(mol3)
        for i, r2 in enumerate(rings2):
            for j, r3 in enumerate(rings3):
                # print(i,j)
                assert r2 != r3

    pass

