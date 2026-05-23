import os
import sys
import torch
import numpy as np
import scipy.optimize
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "model_ddG_3D"))

from model_ddG_3D.model import PretrainModel
from validate_s669 import get_pair_features

ATOM_CA = 1

def run_attack_for_sample(model, wt, mut, true_ddg, device, maxiter=3, popsize=2, k=20):
    with torch.no_grad():
        clean_pred = model(wt, mut).item()

    mut_pos_idx = wt["mut_pos"].argmax().item()
    coords = wt["pos14"][0, :, ATOM_CA, :].cpu().numpy()
    pivot_coord = coords[mut_pos_idx].reshape(1, 3)
    distances = cdist(pivot_coord, coords)[0]
    
    with torch.no_grad():
        model(wt, mut)
    alpha = model.encoder.blocks[-1].saved_alpha[0]
    importance = alpha.sum(dim=(0, 2)).cpu().numpy()
    scores = importance / (distances + 1.0)
    topk_indices = scores.argsort()[::-1][:k].copy()

    wt_emb = wt["dynamic_embedding"].clone().detach().requires_grad_(True)
    wt_grad_input = {k: (v if k != "dynamic_embedding" else wt_emb) for k, v in wt.items()}
    pred = model(wt_grad_input, mut)
    pred.backward()
    noise_basis_vec = wt_emb.grad[0, topk_indices].detach()
    noise_basis_vec = noise_basis_vec / (torch.norm(noise_basis_vec, dim=1, keepdim=True) + 1e-8)

    def objective(x_vars):
        x_vars = x_vars.reshape(k, 4)
        pos_shifts = x_vars[:, :3]
        noise_scales = x_vars[:, 3]

        perturbed_pos14 = wt["pos14"][0].cpu().clone()
        for i, res_idx in enumerate(topk_indices):
            perturbed_pos14[res_idx, :, :] += torch.tensor(pos_shifts[i], dtype=torch.float32)
        new_pair = get_pair_features(perturbed_pos14)

        noise_delta = torch.zeros_like(wt["dynamic_embedding"][0])
        for i, res_idx in enumerate(topk_indices):
            noise_delta[res_idx] += torch.tensor(noise_scales[i]).to(device) * noise_basis_vec[i]

        wt_adv = {key: val.clone() for key, val in wt.items()}
        mut_adv = {key: val.clone() for key, val in mut.items()}
        wt_adv["pos14"] = perturbed_pos14.unsqueeze(0).to(device)
        wt_adv["pair"] = new_pair.unsqueeze(0).to(device)
        wt_adv["dynamic_embedding"] = (wt["dynamic_embedding"][0] + noise_delta).unsqueeze(0).to(device)
        mut_adv["pos14"] = perturbed_pos14.unsqueeze(0).to(device)
        mut_adv["pair"] = new_pair.unsqueeze(0).to(device)
        mut_adv["dynamic_embedding"] = (mut["dynamic_embedding"][0] + noise_delta).unsqueeze(0).to(device)

        with torch.no_grad():
            adv_pred = model(wt_adv, mut_adv).item()
        return -abs(adv_pred - clean_pred)

    def callback(xk, convergence=None):
        if -objective(xk) > 1.25:
            return True
        return False

    bounds = []
    for _ in range(k):
        bounds.extend([(-0.35, 0.35), (-0.35, 0.35), (-0.35, 0.35)])
        bounds.append((-0.1, 0.1))

    res = scipy.optimize.differential_evolution(
        objective, bounds, 
        maxiter=maxiter, popsize=popsize, mutation=0.5, recombination=0.7, workers=1, tol=0, atol=0,
        callback=callback
    )
    
    best_drift = -res.fun
    final_x = res.x.reshape(k, 4)
    pos_shifts = final_x[:, :3]
    
    perturbed_pos = wt["pos14"][0, :, ATOM_CA, :].cpu().clone()
    original_pos = wt["pos14"][0, :, ATOM_CA, :].cpu().clone()
    for i, res_idx in enumerate(topk_indices):
        perturbed_pos[res_idx] += torch.tensor(pos_shifts[i], dtype=torch.float32)
    rmsd = torch.sqrt(torch.mean(torch.sum((perturbed_pos - original_pos)**2, dim=1))).item()

    perturbed_pos14 = wt["pos14"][0].cpu().clone()
    for i, res_idx in enumerate(topk_indices):
        perturbed_pos14[res_idx, :, :] += torch.tensor(pos_shifts[i], dtype=torch.float32)
    new_pair = get_pair_features(perturbed_pos14)
    noise_delta = torch.zeros_like(wt["dynamic_embedding"][0])
    for i, res_idx in enumerate(topk_indices):
        noise_delta[res_idx] += torch.tensor(final_x[i, 3]).to(device) * noise_basis_vec[i]
    
    wt_adv = {key: val.clone() for key, val in wt.items()}
    mut_adv = {key: val.clone() for key, val in mut.items()}
    wt_adv["pos14"] = perturbed_pos14.unsqueeze(0).to(device)
    wt_adv["pair"] = new_pair.unsqueeze(0).to(device)
    wt_adv["dynamic_embedding"] = (wt["dynamic_embedding"][0] + noise_delta).unsqueeze(0).to(device)
    mut_adv["pos14"] = perturbed_pos14.unsqueeze(0).to(device)
    mut_adv["pair"] = new_pair.unsqueeze(0).to(device)
    mut_adv["dynamic_embedding"] = (mut["dynamic_embedding"][0] + noise_delta).unsqueeze(0).to(device)
    with torch.no_grad():
        adv_pred = model(wt_adv, mut_adv).item()

    return clean_pred, adv_pred, best_drift, rmsd

def main():
    parser = argparse.ArgumentParser(description="Stackelberg Game-Theoretic DE Attack.")
    parser.add_argument("--sample", type=str, default="", help="Target a specific mutation ID (e.g. mut_660).")
    parser.add_argument("--num-samples", type=str, default="30", help="Number of benchmark samples to run.")
    parser.add_argument("--max-iter", type=str, default="3", help="DE max iterations.")
    parser.add_argument("--pop-size", type=str, default="2", help="DE population size.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    model_arch_path = os.path.join(PROJECT_ROOT, "model_ddG_3D", "model.pt")
    checkpoint_path = os.path.join(PROJECT_ROOT, "pth", "best_model_s8754.pth")
    
    print("Loading model architecture...", flush=True)
    model = torch.load(model_arch_path, map_location="cpu", weights_only=False)
    print(f"Loading checkpoint weights from {checkpoint_path}...", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.eval().to(device)

    features_dir = os.path.join(PROJECT_ROOT, "S669_features")
    index_csv = os.path.join(features_dir, "index_no_leakage.csv")
    if not os.path.exists(index_csv):
        index_csv = os.path.join(features_dir, "index.csv")

    if not os.path.exists(index_csv):
        print(f"Cannot find S669 features index at {index_csv}. Run validate_s669.py first.")
        sys.exit(1)

    df = pd.read_csv(index_csv, names=["mut_id", "pdb_id", "ddg"])

    max_iter = int(args.max_iter)
    pop_size = int(args.pop_size)

    if args.sample:
                                
        row = df[df["mut_id"] == args.sample]
        if len(row) == 0:
            print(f"Mutation ID {args.sample} not found in index.")
            sys.exit(1)
        row = row.iloc[0]
        pt_path = os.path.join(features_dir, args.sample, "ensemble.pt")
        print(f"Running DE Stackelberg Attack on single sample: {args.sample} ({row['pdb_id']})", flush=True)
        data = torch.load(pt_path, map_location=device)
        wt = {k: v.to(device) for k, v in data["ddG"]["wt"].items() if isinstance(v, torch.Tensor)}
        mut = {k: v.to(device) for k, v in data["ddG"]["mut"].items() if isinstance(v, torch.Tensor)}
        
        mi = 100 if max_iter == 3 else max_iter
        ps = 30 if pop_size == 2 else pop_size
        
        clean_pred, adv_pred, best_drift, rmsd = run_attack_for_sample(model, wt, mut, data["DDG_label"], device, maxiter=mi, popsize=ps)
        print("\n=== Single Sample Results ===")
        print(f"Clean Prediction : {clean_pred:.4f}")
        print(f"Adv Prediction   : {adv_pred:.4f}")
        print(f"Drift            : {best_drift:.4f}")
        print(f"RMSD             : {rmsd:.4f}")
    else:
                       
        num_samples = int(args.num_samples)
        valid_rows = []
        for idx, row in df.iterrows():
            pt_path = os.path.join(features_dir, row["mut_id"], "ensemble.pt")
            if os.path.exists(pt_path):
                valid_rows.append(row)
        
        target_rows = valid_rows[:num_samples]
        print(f"Running Stackelberg Attack Benchmark on {len(target_rows)} samples...", flush=True)

        clean_preds = []
        adv_preds = []
        truths = []
        drifts = []
        RMSDs = []
        successes = 0

        for idx, row in enumerate(target_rows):
            mut_id = row["mut_id"]
            pt_path = os.path.join(features_dir, mut_id, "ensemble.pt")
            data = torch.load(pt_path, map_location=device)
            wt = {k: v.to(device) for k, v in data["ddG"]["wt"].items() if isinstance(v, torch.Tensor)}
            mut = {k: v.to(device) for k, v in data["ddG"]["mut"].items() if isinstance(v, torch.Tensor)}
            true_ddg = data["DDG_label"]

            clean_pred, adv_pred, drift, rmsd = run_attack_for_sample(model, wt, mut, true_ddg, device, maxiter=max_iter, popsize=pop_size)

            clean_preds.append(clean_pred)
            adv_preds.append(adv_pred)
            truths.append(true_ddg)
            drifts.append(drift)
            RMSDs.append(rmsd)
            if drift >= 1.0:
                successes += 1

            print(f"Sample {mut_id:7s} | Clean: {clean_pred:6.3f} | Adv: {adv_pred:6.3f} | Drift: {drift:6.3f} | RMSD: {rmsd:6.3f}", flush=True)

        clean_r, _ = pearsonr(clean_preds, truths)
        adv_r, _ = pearsonr(adv_preds, truths)
        clean_s, _ = spearmanr(clean_preds, truths)
        adv_s, _ = spearmanr(adv_preds, truths)
        asr = (successes / len(target_rows)) * 100

        print("\n=== FINAL BENCHMARK RESULTS ===")
        print(f"Attack Success Rate (ASR) : {asr:.2f}%")
        print(f"Prediction Drift (Avg)    : {np.mean(drifts):.4f} kcal/mol")
        print(f"RMSD (Avg)                : {np.mean(RMSDs):.4f} A")
        print(f"Clean Pearson R           : {clean_r:.4f}")
        print(f"Adversarial Pearson R     : {adv_r:.4f}")
        print(f"Pearson R Decay           : {clean_r - adv_r:.4f}")
        print(f"Clean Spearman Rho        : {clean_s:.4f}")
        print(f"Adversarial Spearman Rho  : {adv_s:.4f}")
        print(f"Spearman Rho Decay        : {clean_s - adv_s:.4f}")

if __name__ == "__main__":
    main()
