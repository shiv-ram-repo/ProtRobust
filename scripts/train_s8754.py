import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "model_ddG_3D"))

from model_ddG_3D.model import PretrainModel

NON_STANDARD_SUBSTITUTIONS = {
         : "GLU", "ASH": "ASP", "CYX": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
         : "ASP", "3AH": "HIS", "5HP": "GLU", "ACL": "ARG", "AGM": "ARG", "AIB": "ALA",
         : "ALA", "ALO": "THR", "ALY": "LYS", "ARM": "ARG", "ASA": "ASP", "ASB": "ASP",
         : "ASP", "ASL": "ASP", "ASQ": "ASP", "AYA": "ALA", "BCS": "CYS", "BHD": "ASP",
         : "THR", "BNN": "ALA", "BUC": "CYS", "BUG": "LEU", "C5C": "CYS", "C6C": "CYS",
         : "CYS", "CCS": "CYS", "CEA": "CYS", "CGU": "GLU", "CHG": "ALA", "CLE": "LEU",
         : "CYS", "CSD": "ALA", "CSO": "CYS", "CSP": "CYS", "CSS": "CYS", "CSW": "CYS",
         : "CYS", "CXM": "MET", "CY1": "CYS", "CY3": "CYS", "CYG": "CYS", "CYM": "CYS",
         : "CYS", "DAH": "PHE", "DAL": "ALA", "DAR": "ARG", "DAS": "ASP", "DCY": "CYS",
         : "GLU", "DGN": "GLN", "DHA": "ALA", "DHI": "HIS", "DIL": "ILE", "DIV": "VAL",
         : "LEU", "DLY": "LYS", "DNP": "ALA", "DPN": "PHE", "DPR": "PRO", "DSN": "SER",
         : "ASP", "DTH": "THR", "DTR": "TRP", "DTY": "TYR", "DVA": "VAL", "EFC": "CYS",
         : "ALA", "FME": "MET", "GGL": "GLU", "GL3": "GLY", "GLZ": "GLY", "GMA": "GLU",
         : "GLY", "HAC": "ALA", "HAR": "ARG", "HIC": "HIS", "HIP": "HIS", "HMR": "ARG",
         : "PHE", "HTR": "TRP", "HYP": "PRO", "IAS": "ASP", "IIL": "ILE", "IYR": "TYR",
         : "LYS", "LLP": "LYS", "LLY": "LYS", "LTR": "TRP", "LYM": "LYS", "LYZ": "LYS",
         : "ALA", "MEN": "ASN", "MHS": "HIS", "MIS": "SER", "MLE": "LEU", "MPQ": "GLY",
         : "GLY", "MSE": "MET", "MVA": "VAL", "NEM": "HIS", "NEP": "HIS", "NLE": "LEU",
         : "LEU", "NLP": "LEU", "NMC": "GLY", "OAS": "SER", "OCS": "CYS", "OMT": "MET",
         : "TYR", "PCA": "GLU", "PEC": "CYS", "PHI": "PHE", "PHL": "PHE", "PR3": "CYS",
         : "ALA", "PTR": "TYR", "PYX": "CYS", "SAC": "SER", "SAR": "GLY", "SCH": "CYS",
         : "CYS", "SCY": "CYS", "SEL": "SER", "SEP": "SER", "SET": "SER", "SHC": "CYS",
         : "LYS", "SMC": "CYS", "SOC": "CYS", "STY": "TYR", "SVA": "SER", "TIH": "ALA",
         : "TRP", "TPO": "THR", "TPQ": "ALA", "TRG": "LYS", "TRO": "TRP", "TYB": "TYR",
         : "TYR", "TYQ": "TYR", "TYS": "TYR", "TYY": "TYR"
}

RESIDUE_SIDECHAIN_POSTFIXES = {
       : ["B"], "R": ["B", "G", "D", "E", "Z", "H1", "H2"], "N": ["B", "G", "D1", "D2"],
       : ["B", "G", "D1", "D2"], "C": ["B", "G"], "E": ["B", "G", "D", "E1", "E2"],
       : ["B", "G", "D", "E1", "E2"], "G": [], "H": ["B", "G", "D1", "D2", "E1", "E2"],
       : ["B", "G1", "G2", "D1"], "L": ["B", "G", "D1", "D2"], "K": ["B", "G", "D", "E", "Z"],
       : ["B", "G", "D", "E"], "F": ["B", "G", "D1", "D2", "E1", "E2", "Z"], "P": ["B", "G", "D"],
       : ["B", "G"], "T": ["B", "G1", "G2"], "W": ["B", "G", "D1", "D2", "E1", "E2", "E3", "Z2", "Z3", "H2"],
       : ["B", "G", "D1", "D2", "E1", "E2", "Z", "H"], "V": ["B", "G1", "G2"]
}

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4

def augmented_three_to_one(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    from Bio.PDB.Polypeptide import protein_letters_3to1
    return protein_letters_3to1[three]

def augmented_three_to_index(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    from Bio.PDB.Polypeptide import three_to_index
    return three_to_index(three)

def augmented_is_aa(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    from Bio.PDB.Polypeptide import is_aa
    return is_aa(three, standard=True)

def get_atom_name_postfix(atom):
    name = atom.get_name()
    if name in ("N", "CA", "C", "O"):
        return name
    if name[-1].isnumeric():
        return name[-2:]
    else:
        return name[-1:]

def get_residue_pos14_coordinates(res):
    pos14 = torch.full([14, 3], float("inf"))
    suffix_to_atom = {get_atom_name_postfix(a): a for a in res.get_atoms()}
    atom_order = ["N", "CA", "C", "O"] + RESIDUE_SIDECHAIN_POSTFIXES[augmented_three_to_one(res.get_resname())]
    for i, atom_suffix in enumerate(atom_order):
        if atom_suffix not in suffix_to_atom:
            continue
        pos14[i, 0], pos14[i, 1], pos14[i, 2] = suffix_to_atom[atom_suffix].get_coord().tolist()
    return pos14

def get_coordinate_features(pdb_file, chain_id):
    from Bio.PDB.PDBParser import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, pdb_file)
    model = list(structure)[0]
    chain = model[chain_id] if chain_id in model else list(model)[0]

    aa, resseq, icode, seq = [], [], [], []
    pos14, pos14_mask = [], []
    seq_this = 0
    seq_1letter = ""
    for res in chain:
        resname = res.get_resname()
        if not augmented_is_aa(resname):
            continue
        if not (res.has_id("CA") and res.has_id("C") and res.has_id("N")):
            continue

        restype = augmented_three_to_index(resname)
        seq_1letter += augmented_three_to_one(resname)
        aa.append(restype)

        pos14_this = get_residue_pos14_coordinates(res)
        pos14.append(pos14_this.nan_to_num(posinf=99999))
        pos14_mask.append(pos14_this.isfinite())

        resseq_this = int(res.get_id()[1])
        icode_this = res.get_id()[2]
        if seq_this == 0:
            seq_this = 1
        else:
            d_resseq = resseq_this - resseq[-1]
            if d_resseq == 0:
                seq_this += 1
            else:
                seq_this += d_resseq
        resseq.append(resseq_this)
        icode.append(icode_this)
        seq.append(seq_this)

    if len(aa) == 0:
        return None, None
    
    result = {
            : torch.LongTensor(aa), 
                : torch.LongTensor(resseq), 
               : "".join(icode), 
             : torch.LongTensor(seq), 
               : torch.stack(pos14).detach().cpu().clone(), 
                    : torch.stack(pos14_mask).detach().cpu().clone()
    }
    return result, seq_1letter

def NormVec(V):
    eps = 1e-10
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= torch.norm(axis_x, dim=-1).unsqueeze(1) + eps
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= torch.norm(axis_z, dim=-1).unsqueeze(1) + eps
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y /= torch.norm(axis_y, dim=-1).unsqueeze(1) + eps
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec

def QuaternionMM(q1, q2):
    a = q1[..., 0] * q2[..., 0] - (q1[..., 1:] * q2[..., 1:]).sum(-1)
    bcd = torch.cross(q2[..., 1:], q1[..., 1:], dim=-1) + q1[..., 0].unsqueeze(-1) * q2[..., 1:] + q2[..., 0].unsqueeze(-1) * q1[..., 1:]
    q = torch.cat([a.unsqueeze(-1), bcd], dim=-1)
    return q

def NormQuaternionMM(q1, q2):
    q = QuaternionMM(q1, q2)
    return q / torch.sqrt((q * q).sum(-1, keepdim=True))

def Rotation2Quaternion(r):
    a = torch.sqrt(r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] + 1) / 2.0
    b = (r[..., 2, 1] - r[..., 1, 2]) / (4 * a)
    c = (r[..., 0, 2] - r[..., 2, 0]) / (4 * a)
    d = (r[..., 1, 0] - r[..., 0, 1]) / (4 * a)
    q = torch.stack([a, b, c, d], dim=-1)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    return q

def NormQuaternion(q):
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    q = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1) * q
    return q

def get_pair_features(pos14):
    L = pos14.shape[0]
    rotation = NormVec(pos14[:, :3, :])
    U, _, V = torch.svd(torch.eye(3).unsqueeze(0).permute(0, 2, 1) @ rotation)
    d = torch.sign(torch.det(U @ V.permute(0, 2, 1)))
    Id = torch.eye(3).repeat(L, 1, 1)
    Id[:, 2, 2] = d
    r = V @ (Id @ U.permute(0, 2, 1))
    q = Rotation2Quaternion(r)
    q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)
    QAll = NormQuaternionMM(q.unsqueeze(1).repeat(1, L, 1), q_1.unsqueeze(0).repeat(L, 1, 1))

    QAll[..., 0][torch.isnan(QAll[..., 0])] = 1.0
    QAll[torch.isnan(QAll)] = 0.0
    QAll = NormQuaternion(QAll)

    xyz_CA = torch.einsum("a b i, a i j -> a b j", pos14[:, ATOM_CA].unsqueeze(0) - pos14[:, ATOM_CA].unsqueeze(1), r)
    return torch.cat([xyz_CA, QAll], dim=-1).detach().cpu().clone()

def safe_norm(x, dim=-1, keepdim=True, eps=1e-8):
    return torch.sqrt(torch.sum(x**2, dim=dim, keepdim=keepdim) + eps)

def NormVec_diff(V):
    axis_x = V[:, 2] - V[:, 1]
    axis_x = axis_x / safe_norm(axis_x)
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z = axis_z / safe_norm(axis_z)
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y = axis_y / safe_norm(axis_y)
    return torch.stack([axis_x, axis_y, axis_z], dim=1)

def NormQuaternion_diff(q):
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-8)
    sign_val = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1).detach()
    return sign_val * q

def get_pair_features_diff(pos14):
    L = pos14.shape[0]
    device = pos14.device
    rotation = NormVec_diff(pos14[:, :3, :])
    mat = (torch.eye(3).to(device).unsqueeze(0).permute(0, 2, 1) @ rotation).cpu().detach()
    U_cpu, _, V_cpu = torch.svd(mat)
    U, V = U_cpu.to(device), V_cpu.to(device)
    d = torch.sign(torch.det(U @ V.permute(0, 2, 1)).detach())
    Id = torch.eye(3).to(device).repeat(L, 1, 1)
    Id[:, 2, 2] = d
    r = V @ (Id @ U.permute(0, 2, 1))
    q = Rotation2Quaternion(r)
    q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)
    QAll = NormQuaternionMM(q.unsqueeze(1).repeat(1, L, 1), q_1.unsqueeze(0).repeat(L, 1, 1))
    QAll = torch.where(torch.isnan(QAll), torch.zeros_like(QAll), QAll)
    QAll = NormQuaternion_diff(QAll)
    xyz_CA = torch.einsum("a b i, a i j -> a b j", pos14[:, ATOM_CA].unsqueeze(0) - pos14[:, ATOM_CA].unsqueeze(1), r)
    return torch.cat([xyz_CA, QAll], dim=-1)

def pgd_inner_diff(model, wt, mt, label, eps_coord=0.35, eps_emb=0.1, steps=5, perturb_coord=True, perturb_emb=True):
    from scipy.spatial.distance import cdist
    import torch.nn.functional as F
    device = label.device
    model.eval()
    
    if perturb_coord:
        m_idx = wt["mut_pos"].long().argmax().item()
        coords = wt["pos14"][0, :, ATOM_CA, :].detach().cpu().numpy()
        target_indices = cdist(coords[m_idx:m_idx+1], coords)[0].argsort()[:20]
        d_pos = torch.zeros((20, 14, 3), device=device, requires_grad=True)
        nn.init.uniform_(d_pos, -eps_coord/10, eps_coord/10)
    else:
        d_pos = None

    if perturb_emb:
        if wt["dynamic_embedding"].shape[0] == 1 and perturb_coord:
            d_emb = torch.zeros((20, 1280), device=device, requires_grad=True)
            nn.init.uniform_(d_emb, -eps_emb/10, eps_emb/10)
        else:
            d_emb = torch.zeros_like(wt["dynamic_embedding"]).uniform_(-eps_emb/10, eps_emb/10).requires_grad_(True)
    else:
        d_emb = None

    params = [p for p in [d_pos, d_emb] if p is not None]
    if len(params) == 0:
        return wt, mt

    optimizer = torch.optim.SGD(params, lr=0.1)
    
    for _ in range(steps):
        p_wt = {k: v.clone() for k, v in wt.items()}
        p_mt = {k: v.clone() for k, v in mt.items()}
        
        pos = wt["pos14"][0].clone() if d_pos is not None else None
        
        if d_pos is not None:
            for i, idx in enumerate(target_indices):
                pos[idx] = pos[idx] + d_pos[i]
            p_wt["pos14"] = pos.unsqueeze(0)
            p_mt["pos14"] = pos.unsqueeze(0)
            p_wt["pair"] = get_pair_features_diff(pos).unsqueeze(0)
            p_mt["pair"] = p_wt["pair"]

        if d_emb is not None:
            if wt["dynamic_embedding"].shape[0] == 1 and d_pos is not None:
                emb_w = wt["dynamic_embedding"][0].clone()
                emb_m = mt["dynamic_embedding"][0].clone()
                for i, idx in enumerate(target_indices):
                    emb_w[idx] = emb_w[idx] + d_emb[i]
                    emb_m[idx] = emb_m[idx] + d_emb[i]
                p_wt["dynamic_embedding"] = emb_w.unsqueeze(0)
                p_mt["dynamic_embedding"] = emb_m.unsqueeze(0)
            else:
                p_wt["dynamic_embedding"] = wt["dynamic_embedding"] + d_emb
                p_mt["dynamic_embedding"] = mt["dynamic_embedding"] + d_emb
                
        loss = -F.mse_loss(model(p_wt, p_mt), label)
        optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            if d_pos is not None:
                d_pos.data = (d_pos - 0.07 * d_pos.grad.sign()).clamp(-eps_coord, eps_coord)
                d_pos.grad.zero_()
            if d_emb is not None:
                d_emb.data = (d_emb - 0.02 * d_emb.grad.sign()).clamp(-eps_emb, eps_emb)
                d_emb.grad.zero_()

    with torch.no_grad():
        final_pos = wt["pos14"][0].clone() if d_pos is not None else wt["pos14"][0]
        if d_pos is not None:
            for i, idx in enumerate(target_indices):
                final_pos[idx] = final_pos[idx] + d_pos[i]
            final_pair = get_pair_features_diff(final_pos)
        else:
            final_pair = wt["pair"][0]

        final_emb_w = wt["dynamic_embedding"][0].clone()
        final_emb_m = mt["dynamic_embedding"][0].clone()
        if d_emb is not None:
            if wt["dynamic_embedding"].shape[0] == 1 and d_pos is not None:
                for i, idx in enumerate(target_indices):
                    final_emb_w[idx] = final_emb_w[idx] + d_emb[i]
                    final_emb_m[idx] = final_emb_m[idx] + d_emb[i]
            else:
                final_emb_w = wt["dynamic_embedding"][0] + d_emb[0]
                final_emb_m = mt["dynamic_embedding"][0] + d_emb[0]

    out_wt = {k: v.clone().detach() for k, v in wt.items()}
    out_mt = {k: v.clone().detach() for k, v in mt.items()}
    out_wt["pos14"] = final_pos.unsqueeze(0).detach()
    out_mt["pos14"] = final_pos.unsqueeze(0).detach()
    out_wt["dynamic_embedding"] = final_emb_w.unsqueeze(0).detach()
    out_mt["dynamic_embedding"] = final_emb_m.unsqueeze(0).detach()
    out_wt["pair"] = final_pair.unsqueeze(0).detach()
    out_mt["pair"] = out_wt["pair"]
    
    return out_wt, out_mt

aa_phy_chem_7_dict = {
       : [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
       : [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
       : [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
       : [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
       : [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
       : [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
       : [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
       : [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
       : [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
       : [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
       : [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
       : [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
       : [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
       : [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
       : [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
       : [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
       : [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
       : [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
       : [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
       : [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]
}

def get_fixed_embedding(seq):
    data = torch.zeros((len(seq), 7), dtype=torch.float32)
    for index, value in enumerate(list(seq)):
        for i in range(7):
            data[index, i] = aa_phy_chem_7_dict[value][i]
    return data.detach().cpu().clone()

def mask_and_unsqueeze_dict(data, mask):
    out = {}
    for k, v in data.items():
        if k in ("fixed_embedding", "dynamic_embedding", "pos14", "atom_mask", "mut_pos"):
            out[k] = v[mask].unsqueeze(0)
        elif k in ("pair"):
            out[k] = v[mask][:, mask].unsqueeze(0)
        else:
            raise ValueError(f"Unknown key: {k}")
    return out

def prepare_s8754_features(csv_path, pdb_dir, out_dir):
                                                       
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)
    index_csv = os.path.join(out_dir, "index.csv")
    if os.path.exists(index_csv):
        os.remove(index_csv)

    print("Loading ESM models...", flush=True)
    model_esm2, alphabet_esm2 = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model_esm2 = model_esm2.eval().to(device)
    batch_converter_esm2 = alphabet_esm2.get_batch_converter()

    esm1v_models = []
    for i in range(1, 6):
        m, a = torch.hub.load("facebookresearch/esm:main", f"esm1v_t33_650M_UR90S_{i}")
        m = m.eval().to(device)
        esm1v_models.append((m, a))

    df = pd.read_csv(csv_path)
    wt_cache = {}
    amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
    amino_acid_dict = {value: index for index, value in enumerate(amino_acid_list)}

    default_pH = 7.0
    default_temp = 2.5

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting S8754 Features"):
        mut_id = f"mut_{idx}"
        mut_folder = os.path.join(out_dir, mut_id)
        os.makedirs(mut_folder, exist_ok=True)

        pdb_id = row['PDB']
        chain_id = row['chain']
        wt_seq_provided = row['wt_seq']
        mut_seq_provided = row['mut_seq']

        if pdb_id not in wt_cache:
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            if not os.path.exists(pdb_file):
                continue
            try:
                coord_res, seq_1letter = get_coordinate_features(pdb_file, chain_id)
            except Exception:
                continue
            if coord_res is None:
                continue

            wt_pos14 = coord_res["pos14"]
            wt_atom_mask = coord_res["pos14_mask"].all(dim=-1)
            wt_pair = get_pair_features(wt_pos14)
            wt_fixed_embedding = get_fixed_embedding(seq_1letter)

            data_esm2 = [("protein", seq_1letter)]
            _, _, batch_tokens_esm2 = batch_converter_esm2(data_esm2)
            with torch.no_grad():
                res_esm2 = model_esm2(batch_tokens_esm2.to(device), repr_layers=[33], return_contacts=False)
                wt_dynamic_embedding = res_esm2["representations"][33][0, 1:-1, :].detach().cpu().clone()

            all_logits_list = []
            for m, a in esm1v_models:
                batch_converter = a.get_batch_converter()
                _, _, batch_tokens = batch_converter([("protein", seq_1letter)])
                with torch.no_grad():
                    token_probs = torch.log_softmax(m(batch_tokens.to(device))["logits"], dim=-1)
                    logits_33 = token_probs[0, 1:-1, :].detach().cpu().clone()
                esm1v_amino_acid_dict = {i: a.get_idx(i) for i in amino_acid_list}
                logits_20 = torch.zeros((logits_33.shape[0], 20))
                for wt_pos, wt_amino_acid in enumerate(seq_1letter):
                    if wt_amino_acid not in esm1v_amino_acid_dict:
                        continue
                    for mut_pos_iter, mut_amino_acid in enumerate(amino_acid_list):
                        logits_20[wt_pos, mut_pos_iter] = logits_33[wt_pos, esm1v_amino_acid_dict[mut_amino_acid]] - logits_33[wt_pos, esm1v_amino_acid_dict[wt_amino_acid]]
                all_logits_list.append(logits_20.unsqueeze(0))

            wt_cache[pdb_id] = {
                                   : wt_dynamic_embedding,
                                 : wt_fixed_embedding,
                      : wt_pair,
                       : wt_pos14,
                           : wt_atom_mask,
                     : seq_1letter,
                        : torch.cat(all_logits_list, dim=0),
                       : torch.ones(len(seq_1letter))
            }

        wt_feats = wt_cache.get(pdb_id)
        if not wt_feats or wt_feats["seq"] != wt_seq_provided:
            continue

        diffs = [i for i in range(min(len(wt_seq_provided), len(mut_seq_provided))) if wt_seq_provided[i] != mut_seq_provided[i]]
        if len(diffs) == 0:
            continue
        mut_char_idx = diffs[0]

        mut_seq_list = list(wt_feats["seq"])
        mut_seq_list[mut_char_idx] = mut_seq_provided[mut_char_idx]
        mut_seq = "".join(mut_seq_list)

        data_esm2 = [("protein", mut_seq)]
        _, _, batch_tokens_esm2 = batch_converter_esm2(data_esm2)
        with torch.no_grad():
            res_esm2 = model_esm2(batch_tokens_esm2.to(device), repr_layers=[33], return_contacts=False)
            mut_dynamic_embedding = res_esm2["representations"][33][0, 1:-1, :].detach().cpu().clone()

        mut_fixed_embedding = get_fixed_embedding(mut_seq)
        pH = row.get('pH', default_pH)
        temperature = row.get('temperature', default_temp * 10) / 10.0
        if pd.isna(pH): pH = default_pH
        if pd.isna(temperature): temperature = default_temp

        length = len(wt_feats["seq"])
        pH_ts = torch.tensor([pH])[None, :].repeat(length, 1)
        temp_ts = torch.tensor([temperature])[None, :].repeat(length, 1)
        plddt_ts = wt_feats["plddt"].unsqueeze(-1)
        wt_lpt = torch.cat((pH_ts, temp_ts, plddt_ts), dim=-1)
        mut_lpt = torch.cat((pH_ts, temp_ts, plddt_ts), dim=-1)

        mut_pos_list = torch.LongTensor([amino_acid_dict.get(i, 0) for i in wt_feats["seq"]]) != torch.LongTensor([amino_acid_dict.get(i, 0) for i in mut_seq])
        mut_pos_list = mut_pos_list.to(torch.float32)

        data = {
                 : {
                    : {
                                       : wt_feats["dynamic_embedding"],
                                     : torch.cat((wt_feats["fixed_embedding"], wt_lpt), dim=-1),
                          : wt_feats["pair"],
                           : wt_feats["pos14"],
                               : wt_feats["atom_mask"],
                             : mut_pos_list
                },
                     : {
                                       : mut_dynamic_embedding,
                                     : torch.cat((mut_fixed_embedding, mut_lpt), dim=-1),
                          : wt_feats["pair"],
                           : wt_feats["pos14"],
                               : wt_feats["atom_mask"],
                             : mut_pos_list
                }
            },
                       : row['DDG']
        }

        coor_CA = wt_feats["pos14"][:, ATOM_CA, :]
        mut_pos_coor_CA = coor_CA[mut_pos_list.to(torch.bool)]
        if mut_pos_coor_CA.shape[0] == 0:
            continue
        diff = mut_pos_coor_CA[0].view(1, 3) - coor_CA.view(-1, 3)
        dist = torch.linalg.norm(diff, dim=-1)
        mask = torch.zeros([dist.shape[0]], dtype=torch.bool)
        m_idx = dist.argsort()[:32]
        mask[m_idx] = True

        data["ddG"]["wt"] = mask_and_unsqueeze_dict(data["ddG"]["wt"], mask)
        data["ddG"]["mut"] = mask_and_unsqueeze_dict(data["ddG"]["mut"], mask)

        torch.save(data, os.path.join(mut_folder, "ensemble.pt"))
        with open(index_csv, "a") as f:
            f.write(f"{mut_id},{pdb_id},{data['DDG_label']}\n")

class GeoStabDataset(Dataset):
    def __init__(self, index_csv):
        df = pd.read_csv(index_csv, names=["mut_id", "pdb_id", "ddg"])
        self.base_dir = os.path.dirname(index_csv)
        valid_rows = []
        for _, row in df.iterrows():
            pt_path = os.path.join(self.base_dir, row['mut_id'], "ensemble.pt")
            if os.path.exists(pt_path):
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mut_id = row['mut_id']
        label = float(row['ddg'])
        pt_path = os.path.join(self.base_dir, mut_id, "ensemble.pt")
        data = torch.load(pt_path, map_location="cpu")
        return data["ddG"]["wt"], data["ddG"]["mut"], label

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def custom_collate(batch):
    wt_list, mut_list, labels = zip(*batch)
    def pad_dict(d_list, keys):
        out = {}
        max_L = max(d[keys[0]].shape[1] for d in d_list)
        B = len(d_list)
        for k in keys:
            tensors = [d[k].squeeze(0) for d in d_list]
            orig_shape = tensors[0].shape
            if len(orig_shape) == 1:
                padded = torch.zeros(B, max_L, dtype=tensors[0].dtype)
                for i, t in enumerate(tensors):
                    padded[i, :t.shape[0]] = t
            elif len(orig_shape) == 2:
                padded = torch.zeros(B, max_L, orig_shape[1], dtype=tensors[0].dtype)
                for i, t in enumerate(tensors):
                    padded[i, :t.shape[0], :] = t
            elif len(orig_shape) == 3:
                if k == "pair":
                    padded = torch.zeros(B, max_L, max_L, orig_shape[2], dtype=tensors[0].dtype)
                    for i, t in enumerate(tensors):
                        l = t.shape[0]
                        padded[i, :l, :l, :] = t
                elif k == "pos14":
                    padded = torch.full((B, max_L, orig_shape[1], orig_shape[2]), 99999.0, dtype=tensors[0].dtype)
                    for i, t in enumerate(tensors):
                        padded[i, :t.shape[0], :, :] = t
            out[k] = padded
        return out
            
    keys = list(wt_list[0].keys())
    batch_wt = pad_dict(wt_list, keys)
    batch_mut = pad_dict(mut_list, keys)
    return batch_wt, batch_mut, torch.tensor(labels, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Train GeoStab ddG predictor on S8754/S669.")
    parser.add_argument("--prepare", action="store_true", help="Run feature preparation first.")
    parser.add_argument("--csv", type=str, default="", help="Path to raw csv (required if preparing).")
    parser.add_argument("--pdb-dir", type=str, default="", help="Path to PDB files directory (required if preparing).")
    parser.add_argument("--epochs", type=str, default="100", help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=str, default="32", help="Training batch size.")
    parser.add_argument("--dataset", type=str, default="s8754", choices=["s8754", "s669"], help="Dataset to train on.")
    parser.add_argument("--robust", type=str, default="none", choices=["none", "pgd", "trades", "squashed"], help="Robust training mode.")
    parser.add_argument("--robust-steps", type=str, default="5", help="Steps for PGD adversary generation.")
    parser.add_argument("--robust-eps-coord", type=str, default="0.35", help="PGD coordinate epsilon.")
    parser.add_argument("--robust-eps-emb", type=str, default="0.1", help="PGD embedding epsilon.")
    args = parser.parse_args()

    if args.dataset == "s669":
        features_dir = os.path.join(PROJECT_ROOT, "S669_features")
        index_csv = os.path.join(features_dir, "index_no_leakage.csv")
        if not os.path.exists(index_csv):
            index_csv = os.path.join(features_dir, "index.csv")
    else:
        features_dir = os.path.join(PROJECT_ROOT, "S8754_features")
        index_csv = os.path.join(features_dir, "index.csv")

    if args.prepare:
        if not args.csv or not args.pdb_dir:
            print("Error: --csv and --pdb-dir must be provided for preparation.")
            sys.exit(1)
        if args.dataset == "s669":
                                                                  
            from validate_s669 import prepare_s669_features
            prepare_s669_features(args.csv, args.pdb_dir, features_dir)
        else:
            prepare_s8754_features(args.csv, args.pdb_dir, features_dir)

    if not os.path.exists(index_csv):
        print(f"Warning: Features not found in {features_dir}.")
        print("To extract features first, run with --prepare --csv <csv_path> --pdb-dir <pdb_dir>")
        print("Using copied features since they already exist in this repository...")
        if not os.path.exists(features_dir):
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_pt_path = os.path.join(PROJECT_ROOT, "model_ddG_3D", "model.pt")
    print(f"Loading base model structure from {model_pt_path}...")
    model = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    
    if args.robust != "none":
        baseline_path = os.path.join(PROJECT_ROOT, "pth", "best_model_s8754.pth")
        if os.path.exists(baseline_path):
            print(f"Loading baseline weights from {baseline_path} for robust fine-tuning...")
            model.load_state_dict(torch.load(baseline_path, map_location="cpu", weights_only=False), strict=False)
        else:
            model.apply(reset_weights)
    else:
        model.apply(reset_weights)
    
    for name, param in model.named_parameters():
        if "encoder" in name or "logits_coef" in name:
            param.requires_grad = True
            
    model = model.to(device)
    
    print("Loading prepared dataset...")
    dataset = GeoStabDataset(index_csv)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    
    num_workers = 0 if device.type == "cpu" else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 if args.robust == "none" else 1e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    if args.robust == "pgd":
        checkpoint_out_path = os.path.join(PROJECT_ROOT, "pth", "robust_model_pgd.pth")
    elif args.robust == "trades":
        checkpoint_out_path = os.path.join(PROJECT_ROOT, "pth", "robust_model_trades.pth")
    elif args.robust == "squashed":
        checkpoint_out_path = os.path.join(PROJECT_ROOT, "pth", "robust_model_squashed.pth")
    else:
        checkpoint_out_path = os.path.join(PROJECT_ROOT, "pth", f"best_model_{args.dataset}.pth")
        
    os.makedirs(os.path.dirname(checkpoint_out_path), exist_ok=True)

    print(f"Starting training loop in mode: {args.robust}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        if args.robust in ["pgd", "trades"]:
                                                                       
            for idx in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{epochs} [Train Robust]"):
                batch_wt, batch_mut, label_tensor = custom_collate([train_dataset[idx]])
                wt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_wt.items()}
                mut = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_mut.items()}
                label_t = label_tensor.to(device)
                
                if args.robust == "pgd":
                    a_wt, a_mut = pgd_inner_diff(model, wt, mut, label_t, 
                                                 eps_coord=float(args.robust_eps_coord), 
                                                 eps_emb=float(args.robust_eps_emb), 
                                                 steps=int(args.robust_steps), 
                                                 perturb_coord=True, perturb_emb=True)
                    model.train()
                    optimizer.zero_grad()
                    preds = model(a_wt, a_mut)
                    loss = criterion(preds, label_t)
                    loss.backward()
                    optimizer.step()
                elif args.robust == "trades":
                    with torch.no_grad():
                        clean_p = model(wt, mut).detach()
                    a_wt, a_mut = pgd_inner_diff(model, wt, mut, clean_p, 
                                                 eps_coord=float(args.robust_eps_coord), 
                                                 eps_emb=float(args.robust_eps_emb), 
                                                 steps=int(args.robust_steps), 
                                                 perturb_coord=True, perturb_emb=True)
                    model.train()
                    optimizer.zero_grad()
                    loss = criterion(model(wt, mut), label_t) + 1.0 * criterion(model(a_wt, a_mut), clean_p)
                    loss.backward()
                    optimizer.step()
                    
                train_loss += loss.item()
            train_loss /= len(train_dataset)
            
        elif args.robust == "squashed":
                                                       
            for batch_idx, (wt, mut, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train Squashed]")):
                for k in wt: wt[k] = wt[k].to(device)
                for k in mut: mut[k] = mut[k].to(device)
                labels = labels.to(device)
                
                model.eval()
                with torch.no_grad():
                    clean_p = model(wt, mut).detach()
                
                a_wt, a_mut = pgd_inner_diff(model, wt, mut, clean_p, 
                                             eps_coord=float(args.robust_eps_coord), 
                                             eps_emb=float(args.robust_eps_emb), 
                                             steps=int(args.robust_steps), 
                                             perturb_coord=False, perturb_emb=True)
                
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(wt, mut), labels) + 1.0 * criterion(model(a_wt, a_mut), clean_p)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * labels.size(0)
            train_loss /= len(train_dataset)
            
        else:
                               
            for batch_idx, (wt, mut, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
                for k in wt: wt[k] = wt[k].to(device)
                for k in mut: mut[k] = mut[k].to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                preds = model(wt, mut)
                loss = criterion(preds, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * labels.size(0)
            train_loss /= len(train_dataset)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (wt, mut, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                for k in wt: wt[k] = wt[k].to(device)
                for k in mut: mut[k] = mut[k].to(device)
                labels = labels.to(device)
                
                preds = model(wt, mut)
                loss = criterion(preds, labels)
                val_loss += loss.item() * labels.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss! Saving checkpoint to {checkpoint_out_path}...")
            torch.save(model.state_dict(), checkpoint_out_path)

if __name__ == "__main__":
    main()
