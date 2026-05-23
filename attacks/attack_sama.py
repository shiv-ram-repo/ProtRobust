import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "model_ddG_3D"))

from model_ddG_3D.model import PretrainModel
from validate_s669 import Rotation2Quaternion, NormQuaternionMM

ATOM_CA = 1

def NormQuaternion_diff(q):
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-8)
    sign_val = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1).detach()
    q = sign_val * q
    return q

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
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec

def get_pair_features_diff(pos14, rank=0, t=0):
    L = pos14.shape[0]
    device = pos14.device
    rotation = NormVec_diff(pos14[:, :3, :])
                                                                                                                             
    mat = (torch.eye(3).to(device).unsqueeze(0).permute(0, 2, 1) @ rotation).cpu().detach()
    if torch.isnan(mat).any() or torch.isinf(mat).any():
        print(f"Sample {rank} | Step {t} | INVALID MATRIX GENERATED BEFORE SVD! NA/INF detected.", flush=True)
    U_cpu, _, V_cpu = torch.svd(mat)
    U = U_cpu.to(device)
    V = V_cpu.to(device)
    
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

class PGDExplorationAgent:
    def __init__(self, k_rad, device, lr_pos=0.05, lr_emb=0.05):
        self.k_rad = k_rad
        self.device = device
        self.lr_pos = lr_pos
        self.lr_emb = lr_emb
        self.sensitivity_map = torch.zeros(k_rad, device=device)
        self.prev_payoff = -float("inf")
        self.stagnant_counter = 0

    def choose_action(self, current_payoff):
        if current_payoff > self.prev_payoff:
            self.stagnant_counter = 0
            action = "G"
        else:
            self.stagnant_counter += 1
            if self.stagnant_counter >= 3:
                action = "J"
            else:
                action = "G"
        self.prev_payoff = current_payoff
        return action

    def apply_exploration_jump(self, A_pos, A_emb):
        with torch.no_grad():
            jump_scale = 0.1 * (1.0 + self.sensitivity_map.unsqueeze(-1).unsqueeze(-1))
            A_pos += torch.randn_like(A_pos) * jump_scale
            A_emb += torch.randn_like(A_emb) * 0.1
        return A_pos, A_emb

    def update_state(self, pos_grad, emb_grad):
        with torch.no_grad():
            grad_norm = torch.norm(pos_grad, dim=(1, 2))
            self.sensitivity_map = 0.9 * self.sensitivity_map + 0.1 * grad_norm

def run_sama_attack(model, wt, mut, device, episodes=5, steps_per_episode=10, reg=0.05, k_rad=10, lr_pos=0.05, lr_emb=0.05, verbose=False, trajectory_list=None):
    with torch.no_grad():
        clean_pred = model(wt, mut).item()
        
    mut_pos_idx = wt["mut_pos"].argmax().item()
    coords = wt["pos14"][0, :, ATOM_CA, :].cpu().numpy()
    pivot = coords[mut_pos_idx].reshape(1, 3)
    dist = cdist(pivot, coords)[0]
    topk_indices = dist.argsort()[:k_rad].copy()

    agent = PGDExplorationAgent(k_rad, device, lr_pos=lr_pos, lr_emb=lr_emb)
    A_pos = torch.zeros((k_rad, 14, 3), device=device, dtype=torch.float32, requires_grad=True)
    A_emb = torch.zeros((k_rad, 1280), device=device, dtype=torch.float32, requires_grad=True)

    current_best_drift = 0.0
    global_step = 0

    for ep in range(episodes):
        for t in range(steps_per_episode):
            if A_pos.grad is not None: A_pos.grad.zero_()
            if A_emb.grad is not None: A_emb.grad.zero_()

            adv_pos14 = wt["pos14"][0].clone()
            for i, res_idx in enumerate(topk_indices):
                adv_pos14[res_idx] = adv_pos14[res_idx] + A_pos[i]

            adv_wt_emb = wt["dynamic_embedding"][0].clone()
            adv_mut_emb = mut["dynamic_embedding"][0].clone()
            for i, res_idx in enumerate(topk_indices):
                adv_wt_emb[res_idx] = adv_wt_emb[res_idx] + A_emb[i]
                adv_mut_emb[res_idx] = adv_mut_emb[res_idx] + A_emb[i]

            adv_wt = {k: v.clone() for k, v in wt.items()}
            adv_mut = {k: v.clone() for k, v in mut.items()}
            adv_wt["pos14"] = adv_pos14.unsqueeze(0)
            adv_wt["pair"] = get_pair_features_diff(adv_pos14).unsqueeze(0)
            adv_wt["dynamic_embedding"] = adv_wt_emb.unsqueeze(0)
            adv_mut["pos14"] = adv_pos14.unsqueeze(0) 
            adv_mut["pair"] = adv_wt["pair"]
            adv_mut["dynamic_embedding"] = adv_mut_emb.unsqueeze(0)
            
            pred = model(adv_wt, adv_mut)
            drift = torch.abs(pred - clean_pred)
            
            payoff = -drift + reg * (torch.sum(A_pos**2) + torch.sum(A_emb**2))
            payoff.backward()
            
            strategy = agent.choose_action(-payoff.item())
            
            if trajectory_list is not None:
                trajectory_list.append({
                          : global_step,
                           : drift.item(),
                              : strategy
                })
            
            with torch.no_grad():
                if strategy == "G":
                    A_pos -= agent.lr_pos * torch.clamp(A_pos.grad, -1.0, 1.0)
                    A_emb -= agent.lr_emb * torch.clamp(A_emb.grad, -1.0, 1.0)
                elif strategy == "J":
                    A_pos, A_emb = agent.apply_exploration_jump(A_pos, A_emb)
                
                agent.update_state(A_pos.grad, A_emb.grad)
            
            if drift.item() > current_best_drift:
                current_best_drift = drift.item()
            
            global_step += 1

        with torch.no_grad():
            diff = adv_pos14[topk_indices] - wt["pos14"][0, topk_indices]
            rmsd = torch.sqrt((diff**2).sum(dim=-1).mean())
            if rmsd > 0.05:
                A_pos *= 0.5

    return clean_pred, current_best_drift, rmsd.item()

def main():
    parser = argparse.ArgumentParser(description="Sama-Agentic AI PGD-Exploration Attack.")
    parser.add_argument("--sample", type=str, default="", help="Target a specific mutation ID (e.g. mut_0).")
    parser.add_argument("--num-samples", type=str, default="30", help="Number of benchmark samples to run.")
    parser.add_argument("--episodes", type=str, default="5", help="Number of SAMA episodes.")
    parser.add_argument("--steps", type=str, default="10", help="Steps per episode.")
    parser.add_argument("--reg", type=str, default="0.05", help="Regularization penalty.")
    parser.add_argument("--k-rad", type=str, default="10", help="Attack radius size.")
    parser.add_argument("--lr-pos", type=str, default="0.05", help="Coordinate learning rate.")
    parser.add_argument("--lr-emb", type=str, default="0.05", help="Embedding learning rate.")
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

    episodes = int(args.episodes)
    steps = int(args.steps)
    reg = float(args.reg)
    k_rad = int(args.k_rad)
    lr_pos = float(args.lr_pos)
    lr_emb = float(args.lr_emb)

    if args.sample:
        row = df[df["mut_id"] == args.sample]
        if len(row) == 0:
            print(f"Mutation ID {args.sample} not found in index.")
            sys.exit(1)
        row = row.iloc[0]
        pt_path = os.path.join(features_dir, args.sample, "ensemble.pt")
        print(f"Running SAMA Attack on single sample: {args.sample} ({row['pdb_id']})", flush=True)
        data = torch.load(pt_path, map_location=device)
        wt = {k: v.to(device) for k, v in data["ddG"]["wt"].items() if isinstance(v, torch.Tensor)}
        mut = {k: v.to(device) for k, v in data["ddG"]["mut"].items() if isinstance(v, torch.Tensor)}
        
        trajectory_points = []
        clean_pred, best_drift, rmsd = run_sama_attack(
            model, wt, mut, device, episodes=episodes, steps_per_episode=steps, reg=reg, k_rad=k_rad,
            lr_pos=lr_pos, lr_emb=lr_emb, verbose=True, trajectory_list=trajectory_points
        )
        
        print("\n=== Single Sample Results ===")
        print(f"Clean Prediction : {clean_pred:.4f}")
        print(f"Best Drift       : {best_drift:.4f}")
        print(f"Final RMSD       : {rmsd:.4f}")

        if trajectory_points:
            print("\nGenerating PGD-Exploration Trajectory Plot...", flush=True)
            steps_arr = [p['step'] for p in trajectory_points]
            drifts_arr = [p['drift'] for p in trajectory_points]
            strats_arr = [p['strategy'] for p in trajectory_points]
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel('Game Time Step (t)')
            ax1.set_ylabel('Adversarial Drift (kcal/mol)', color='tab:blue')
            ax1.plot(steps_arr, drifts_arr, color='tab:blue', linewidth=2, label='Prediction Drift')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Agent Strategy (0:G, 1:J)', color='tab:red')
            strat_mapped = [0 if s == "G" else 1 for s in strats_arr]
            ax2.scatter(steps_arr, strat_mapped, color='tab:red', alpha=0.6, label='Strategy Jumps (J)')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['G (Flow)', 'J (Jump)'])
            ax2.tick_params(axis='y', labelcolor='tab:red')

            plt.title(f'SAMA Reasoning Trajectory: Strategic Switching ({args.sample})')
            fig.tight_layout()
            
            plot_path = os.path.join(PROJECT_ROOT, "logs", f"sama_trajectory_{args.sample}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Trajectory plot saved to {plot_path}", flush=True)
    else:
        num_samples = int(args.num_samples) if args.num_samples.lower() != 'all' else len(df)
        valid_rows = []
        for idx, row in df.iterrows():
            pt_path = os.path.join(features_dir, row["mut_id"], "ensemble.pt")
            if os.path.exists(pt_path):
                valid_rows.append(row)
        
        target_rows = valid_rows[:num_samples]
        print(f"Running SAMA Attack Benchmark on {len(target_rows)} samples...", flush=True)

        all_clean_preds = []
        all_drifts = []

        for idx, row in enumerate(target_rows):
            mut_id = row["mut_id"]
            pt_path = os.path.join(features_dir, mut_id, "ensemble.pt")
            data = torch.load(pt_path, map_location=device)
            wt = {k: v.to(device) for k, v in data["ddG"]["wt"].items() if isinstance(v, torch.Tensor)}
            mut = {k: v.to(device) for k, v in data["ddG"]["mut"].items() if isinstance(v, torch.Tensor)}

            clean_pred, best_drift, rmsd = run_sama_attack(
                model, wt, mut, device, episodes=episodes, steps_per_episode=steps, reg=reg, k_rad=k_rad,
                lr_pos=lr_pos, lr_emb=lr_emb, verbose=False
            )

            all_clean_preds.append(clean_pred)
            all_drifts.append(best_drift)

            if idx < 5:
                print(f"Sample {mut_id:7s} | Clean: {clean_pred:6.3f} | Best Drift: {best_drift:6.3f} | Final RMSD: {rmsd:6.4f}", flush=True)
            elif idx == 5:
                print("...", flush=True)

        drifts_arr = np.array(all_drifts)
        print(f"\n=== FINAL BENCHMARK RESULTS ===")
        print(f"Total Samples Processed   : {len(target_rows)}")
        print(f"Prediction Drift (Avg)    : {drifts_arr.mean():.4f} kcal/mol")
        print(f"Max Adversarial Deviation : {drifts_arr.max():.4f} kcal/mol")

if __name__ == "__main__":
    main()
