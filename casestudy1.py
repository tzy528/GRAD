# main_b.py
# Version B: KEEP HALO on MLP embeddings, add robust prototype residual auxiliary
# Usage:
#   python main_b.py --data_dir ./datasets/ --dataset YelpChi.mat --device cuda:0

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import time
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected


import dgl
from dgl.data.utils import load_graphs
import networkx as nx

# -----------------------
# args & defaults
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_dir', type=str, default='./datasets/')
parser.add_argument('--dataset', type=str, default='Amazon')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--output_dim', type=int, default=64)
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--alpha_proto', type=float, default=0.25, help='weight of prototype auxiliary score in final score')
parser.add_argument('--alpha_align', type=float, default=0.2, help='weight of asymmetric alignment loss')
parser.add_argument('--alpha_rank', type=float, default=2.0, help='weight of ranking loss')
parser.add_argument('--protoweight', type=float, default=0.2, help='prototype update weight')
parser.add_argument('--proto_topk', type=int, default=200)
parser.add_argument('--proto_update_interval', type=int, default=5)
parser.add_argument('--halo_chunk', type=int, default=300, help='chunk size for halo update')
parser.add_argument('--device_fallback', action='store_true')
args = parser.parse_args()

# seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
device = torch.device(args.device if (torch.cuda.is_available() or args.device_fallback) else 'cpu')
# device = 'cpu'
print("Device:", device)

# -----------------------
# data loader (robust to common .mat keys)
# -----------------------
def load_data(data_dir, dataset_name):
    if dataset_name in ['AmazonFull', 'YelpChiFull']:
        path = os.path.join(data_dir, dataset_name)

        # 加载DGL图
        graph_list, _ = load_graphs(path)
        graph = graph_list[0]

        # 从DGL图中提取数据
        features = graph.ndata["feature"]
        labels = graph.ndata["label"].cpu().numpy()

        # 转换为无向图并移除自环
        graph = dgl.to_bidirected(graph)
        graph = graph.remove_self_loop()

        # 将DGL图转换为networkx，再转换为PyG格式
        nx_graph = dgl.to_networkx(graph)
        adj = nx.to_scipy_sparse_array(nx_graph)

        # 归一化特征
        if sp.issparse(features):
            features = features.toarray()
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # 转换为PyG所需的格式
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_index = to_undirected(edge_index)

        # 转换为torch张量
        features = torch.FloatTensor(features).to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.float().to(device) if edge_weight is not None else None
        labels = torch.LongTensor([1 if l > 0 else 0 for l in labels]).to(device)

        num_nodes = features.size(0)  # or labels.size(0)
        num_edges_dir = edge_index.size(1)  # edge_index 是 [2, E]，这里的 E 是“列数”

        # 统计无向边数（把(i,j)和(j,i)视为同一条，并去重）
        row, col = edge_index[0], edge_index[1]
        u = torch.minimum(row, col)
        v = torch.maximum(row, col)
        num_edges_undir = torch.unique(torch.stack([u, v], dim=0), dim=1).size(1)
        feat_dim = features.size(1)

        num_pos = (labels == 1).sum().item()

        print(f"#Nodes: {num_nodes}")
        print(f"#Edges (directed in edge_index): {num_edges_dir}")
        print(f"#Edges (undirected unique): {num_edges_undir}")
        print(f"#Pos labels (label==1): {num_pos}")
        print(f"#Feature dim: {feat_dim}")

    else:
        path = os.path.join(data_dir, dataset_name+'.mat')
        mat = sio.loadmat(path)
        # adjacency
        if 'Network' in mat:
            adj = mat['Network']
        elif 'net' in mat:
            adj = mat['net']
        elif 'homo' in mat:
            adj = mat['homo']
        elif 'A' in mat:
            adj = mat['A']
        else:
            raise ValueError("Adjacency not found in mat keys: " + ", ".join(list(mat.keys())))
        adj = sp.coo_matrix(adj)

        # features
        if 'Attributes' in mat:
            feat = mat['Attributes']
        elif 'features' in mat:
            feat = mat['features']
        elif 'attr' in mat:
            feat = mat['attr']
        elif 'X' in mat:
            feat = mat['X']
        else:
            raise ValueError("Features not found in mat keys: " + ", ".join(list(mat.keys())))
        if sp.issparse(feat):
            feat = feat.toarray()

        # labels
        if 'Label' in mat:
            labels = mat['Label'].flatten()
        elif 'label' in mat:
            labels = mat['label'].flatten()
        elif 'labels' in mat:
            labels = mat['labels'].flatten()
        else:
            raise ValueError("Labels not found in mat keys: " + ", ".join(list(mat.keys())))
        labels = np.array([1 if l > 0 else 0 for l in labels], dtype=np.int64)

        # normalize features
        scaler = StandardScaler()
        feat = scaler.fit_transform(feat)

        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_index = to_undirected(edge_index)

        features = torch.FloatTensor(feat).to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.float().to(device) if edge_weight is not None else None
        labels = torch.LongTensor(labels).to(device)
        num_pos = (labels == 1).sum().item()
        print(f"Loaded: nodes={features.size(0)} feat_dim={features.size(1)} edges={edge_index.size(1)} fake={num_pos}")
    return features, edge_index, edge_weight, labels

def split_dataset(labels, train_ratio=0.7, val_ratio=0.1, seed=42):
    N = len(labels)
    idx = np.arange(N)
    labels_np = labels.cpu().numpy()
    train_val_idx, test_idx = train_test_split(idx, test_size=1-train_ratio, stratify=labels_np, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, stratify=labels_np[train_val_idx], random_state=seed)
    train_mask = torch.zeros(N, dtype=torch.bool).to(device)
    val_mask = torch.zeros(N, dtype=torch.bool).to(device)
    test_mask = torch.zeros(N, dtype=torch.bool).to(device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    print(f"Split: train={train_mask.sum().item()}, val={val_mask.sum().item()}, test={test_mask.sum().item()}")
    return train_mask, val_mask, test_mask

# -----------------------
# HALO but on MLP embeddings (chunked to fit GPU)
# HALO(xi,xj) = ||xhat_i - xhat_j|| / sqrt(||xhat_i||^2 + ||xhat_j||^2 + eps),
# where xhat_i = abs(xi-xj) * xi (here xi are mlp embeddings)
# -----------------------
def compute_HALO_emb(embeddings, batch_indices=None, epsilon=1e-8, chunk_size=2000):
    # embeddings: torch [N, d]
    if batch_indices is not None:
        X = embeddings[batch_indices]
        b = X.size(0)
        diff = torch.abs(X.unsqueeze(1) - X.unsqueeze(0))  # [b,b,d]
        xi_hat = diff * X.unsqueeze(1)
        xj_hat = diff * X.unsqueeze(0)
        numerator = torch.norm(xi_hat - xj_hat, dim=2)
        xi_norm_sq = torch.sum(xi_hat**2, dim=2)
        xj_norm_sq = torch.sum(xj_hat**2, dim=2)
        denom = torch.sqrt(xi_norm_sq + xj_norm_sq + epsilon)
        halo = numerator / (denom + epsilon)
        mask = (1.0 - torch.eye(b, device=embeddings.device))
        masked = halo * mask
        mean_halo = masked.sum(dim=1) / (b - 1 + 1e-12)
        return mean_halo
    else:
        N = embeddings.size(0)
        if N <= chunk_size:
            return compute_HALO_emb(embeddings, torch.arange(N, device=embeddings.device), epsilon=epsilon)
        else:
            out = torch.zeros(N, device=embeddings.device)
            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)
                out[i:j] = compute_HALO_emb(embeddings, torch.arange(i, j, device=embeddings.device), epsilon=epsilon)
            return out

# -----------------------
# Model: MLP (attribute), GCN (structure), prototypes
# -----------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.conv1 = GCNConv(in_dim, out_dim)
        else:
            self.conv1 = GCNConv(in_dim, hid_dim)
            self.convs = nn.ModuleList([GCNConv(hid_dim, hid_dim) for _ in range(num_layers-2)])
            self.conv_last = GCNConv(hid_dim, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index, edge_weight=None):
        if self.num_layers == 1:
            return self.conv1(x, edge_index, edge_weight)
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_weight))
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv_last(h, edge_index, edge_weight)
        return h

class ModelB(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, gcn_layers=2, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        # GCN takes raw features so structural branch is independent
        self.gcn = SimpleGCN(in_dim=in_dim, hid_dim=hidden_dim, out_dim=out_dim, num_layers=gcn_layers, dropout=dropout)
        # prototypes in residual (mlp residual) space
        self.normal_proto = nn.Parameter(F.normalize(torch.randn(out_dim) * 0.5 + 0.5, dim=0))
        self.anom_proto = nn.Parameter(F.normalize(torch.randn(out_dim) * 0.5 - 0.5, dim=0))
        self.out_dim = out_dim

    def forward_full(self, raw_features, edge_index, edge_weight=None):
        # compute mlp embedding based on raw features
        mlp_emb = self.mlp(raw_features)         # [N, D]
        gcn_emb = self.gcn(raw_features, edge_index, edge_weight)  # [N, D]
        return mlp_emb, gcn_emb

    def neighbor_mean_from_emb(self, emb, edge_index):
        src, dst = edge_index[0], edge_index[1]
        device_local = emb.device
        neighbor_sum = torch.zeros_like(emb)
        neighbor_count = torch.zeros(emb.size(0), device=device_local)
        neighbor_sum = neighbor_sum.index_add(0, dst, emb[src])
        neighbor_count = neighbor_count.index_add(0, dst, torch.ones_like(dst.float(), device=device_local))
        neighbor_count_safe = neighbor_count.clone(); neighbor_count_safe[neighbor_count_safe==0] = 1.0
        neighbor_mean = neighbor_sum / neighbor_count_safe.unsqueeze(1)
        return neighbor_mean

    def final_scores(self, mlp_emb, gcn_emb, edge_index, alpha_proto=0.25):
        # compute neighbor mean of mlp_emb
        neighbor_mean = self.neighbor_mean_from_emb(mlp_emb, edge_index)
        # HUGE main: local inconsistency = 1 - cos(mlp, neighbor_mean)
        mlp_norm = F.normalize(mlp_emb, dim=1)
        neigh_norm = F.normalize(neighbor_mean, dim=1)
        local_incons = 1.0 - torch.sum(mlp_norm * neigh_norm, dim=1)
        local_incons = (local_incons - local_incons.mean()) / (local_incons.std() + 1e-8)
        # prototype residual: residual = mlp_emb - neighbor_mean
        residual = mlp_emb - neighbor_mean
        r_norm = F.normalize(residual, dim=1)
        pn = F.normalize(self.normal_proto.unsqueeze(0), dim=1)
        pa = F.normalize(self.anom_proto.unsqueeze(0), dim=1)
        sim_anom = (r_norm * pa).sum(dim=1)
        sim_norm = (r_norm * pn).sum(dim=1)
        proto_score = sim_anom - sim_norm
        proto_score = (proto_score - proto_score.mean()) / (proto_score.std() + 1e-8)
        final = (1.0 - alpha_proto) * local_incons + alpha_proto * proto_score
        return final, local_incons, proto_score, neighbor_mean, residual

# -----------------------
# losses
# -----------------------
def ranking_loss_pairwise(pred_scores, halo_scores, margin=0.2):
    # choose top-k and bottom-k from halo_scores (halo computed on mlp embeddings)
    N = len(halo_scores)
    k = max(5, int(0.15 * N))
    halo_std = (halo_scores - halo_scores.mean()) / (halo_scores.std() + 1e-8)
    _, top_idx = torch.topk(halo_std, k)
    _, bottom_idx = torch.topk(-halo_std, k)
    pos = pred_scores[top_idx]
    neg = pred_scores[bottom_idx]
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    loss = F.relu(margin - diff)
    if loss.numel() == 0:
        return torch.tensor(0.0, device=pred_scores.device)
    return loss.mean()

def asymmetric_alignment_loss(mlp_emb, gcn_emb, edge_index):
    # compute edge-wise cos similarity
    src, dst = edge_index[0], edge_index[1]
    mlp_n = F.normalize(mlp_emb, dim=1)
    gcn_n = F.normalize(gcn_emb, dim=1)
    s_mlp = (mlp_n[src] * mlp_n[dst]).sum(dim=1)  # [E]
    s_gcn = (gcn_n[src] * gcn_n[dst]).sum(dim=1)
    N = mlp_emb.size(0)
    mean_mlp = torch.zeros(N, device=mlp_emb.device)
    mean_gcn = torch.zeros_like(mean_mlp)
    cnt = torch.zeros(N, device=mlp_emb.device)
    mean_mlp = mean_mlp.index_add(0, dst, s_mlp)
    mean_gcn = mean_gcn.index_add(0, dst, s_gcn)
    cnt = cnt.index_add(0, dst, torch.ones_like(dst.float(), device=mlp_emb.device))
    cnt_safe = cnt.clone(); cnt_safe[cnt_safe==0] = 1.0
    mean_mlp = mean_mlp / cnt_safe
    mean_gcn = mean_gcn / cnt_safe
    var_mlp = torch.zeros(N, device=mlp_emb.device)
    var_gcn = torch.zeros(N, device=mlp_emb.device)
    var_mlp = var_mlp.index_add(0, dst, s_mlp**2)
    var_gcn = var_gcn.index_add(0, dst, s_gcn**2)
    var_mlp = var_mlp / cnt_safe - mean_mlp**2
    var_gcn = var_gcn / cnt_safe - mean_gcn**2
    return F.mse_loss(mean_mlp, mean_gcn) + F.mse_loss(var_mlp, var_gcn)

# -----------------------
# evaluation
# -----------------------
def evaluate(model, features, edge_index, edge_weight, labels, mask):
    model.eval()
    with torch.no_grad():
        mlp_emb, gcn_emb = model.forward_full(features, edge_index, edge_weight)
        final, local_incons, proto_score, _, _ = model.final_scores(mlp_emb, gcn_emb, edge_index, alpha_proto=args.alpha_proto)
        s = final[mask].detach().cpu().numpy()
        y = labels[mask].detach().cpu().numpy()
        if (y == 1).sum() > 0 and (y == 0).sum() > 0:
            if s[y == 1].mean() < s[y == 0].mean():
                s = -s
        try:
            auc = roc_auc_score(y, s)
            ap = average_precision_score(y, s)
        except Exception:
            auc, ap = 0.5, 0.5
        return auc, ap

# -----------------------
# training loop
# -----------------------
def train(model, features, edge_index, edge_weight, labels, train_mask, val_mask):
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    best_val = 0.0
    best_epoch = 0
    patience_ctr = 0


    train_idx = train_mask.nonzero().squeeze().cpu().numpy().tolist()
    N = features.size(0)

    # Precompute neighbor indices info if needed (we recompute neighbor means inside model)
    # main loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        # forward full embeddings (needed for alignment & halo)
        mlp_full, gcn_full = model.forward_full(features, edge_index, edge_weight)

        # compute halo on mlp embeddings (this is HALO using MLP emb, same space as residual)
        halo_full = compute_HALO_emb(mlp_full, None, epsilon=1e-8, chunk_size=args.halo_chunk).detach()

        # update prototypes periodically using halo_full selection on residuals
        if epoch % args.proto_update_interval == 0 or epoch == 1:
            with torch.no_grad():
                neighbor_mean_all = model.neighbor_mean_from_emb(mlp_full, edge_index)
                residual_all = mlp_full - neighbor_mean_all
                sorted_idx = torch.argsort(halo_full)  # low -> similar to neighbors, high -> heterophilic
                topk = min(args.proto_topk, N//4)
                normal_idx = sorted_idx[:topk]
                anom_idx = sorted_idx[-topk:]
                new_norm = residual_all[normal_idx].mean(dim=0)
                new_anom = residual_all[anom_idx].mean(dim=0)
                # soft-update prototypes (in residual space)
                model.normal_proto.data = F.normalize(args.protoweight * model.normal_proto.data + (1-args.protoweight) * new_norm.data, dim=0)
                model.anom_proto.data = F.normalize(args.protoweight * model.anom_proto.data + (1-args.protoweight) * new_anom.data, dim=0)

        # sample batch nodes for ranking/proto losses
        bsize = min(args.batch_size, len(train_idx))
        batch_nodes = np.random.choice(train_idx, size=bsize, replace=False)
        batch_nodes = torch.tensor(batch_nodes, device=device, dtype=torch.long)

        optimizer.zero_grad()
        # Use full embeddings to compute alignment and final scores
        mlp_full, gcn_full = model.forward_full(features, edge_index, edge_weight)
        final_full, local_inst_full, proto_score_full, neighbor_mean_full, residual_full = model.final_scores(mlp_full, gcn_full, edge_index, alpha_proto=args.alpha_proto)

        # get batch views
        mlp_b = mlp_full[batch_nodes]
        gcn_b = gcn_full[batch_nodes]
        halo_b = halo_full[batch_nodes].to(device)
        local_inst_b = local_inst_full[batch_nodes]
        proto_b = proto_score_full[batch_nodes]

        # pred_for_rank uses local_inst primarily + proto secondary (same as final composition)
        pred_for_rank = (1.0 - args.alpha_proto) * local_inst_b + args.alpha_proto * proto_b

        # losses
        l_rank = ranking_loss_pairwise(pred_for_rank, halo_b, margin=args.margin)
        l_align = asymmetric_alignment_loss(mlp_full, gcn_full, edge_index)  # full graph alignment
        # prototype alignment: pull residuals to prototypes with soft weights from halo_full
        halo_std_full = (halo_full - halo_full.mean()) / (halo_full.std() + 1e-8)
        prob_anom_full = torch.sigmoid(3.0 * halo_std_full.to(device))
        dist_to_norm = torch.norm(residual_full - model.normal_proto.unsqueeze(0), dim=1)
        dist_to_anom = torch.norm(residual_full - model.anom_proto.unsqueeze(0), dim=1)
        l_proto_align = ((1 - prob_anom_full) * dist_to_norm + prob_anom_full * dist_to_anom).mean()
        # prototype separation
        cos_sim = F.cosine_similarity(model.normal_proto.unsqueeze(0), model.anom_proto.unsqueeze(0)).squeeze()
        l_proto_sep = F.relu(cos_sim - (-0.2)) + F.relu(0.8 - torch.norm(model.normal_proto - model.anom_proto))

        total_loss = args.alpha_rank * l_rank + args.alpha_align * l_align + 0.6 * l_proto_align + 0.1 * l_proto_sep

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        # eval
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            val_auc, val_ap = evaluate(model, features, edge_index, edge_weight, labels, val_mask)
            print(f"Epoch {epoch:03d} loss={total_loss.item():.4f} val_auc={val_auc:.4f} val_ap={val_ap:.4f} "
                  f"l_rank={l_rank.item():.4f} l_align={l_align.item():.4f} l_proto={l_proto_align.item():.4f} time={time.time()-t0:.4f}s")
            if val_auc > best_val + 1e-4:
                best_val = val_auc
                best_epoch = epoch
                patience_ctr = 0
                torch.save(model.state_dict(), "best_model_b.pth")
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print("Early stopping.")
                    break

    # load best
    if os.path.exists("best_model_b.pth"):
        model.load_state_dict(torch.load("best_model_b.pth", map_location=device))
    return model, best_val, best_epoch

##############################################
# 可视化 + 单节点解释  (插入到 main() 结尾前)
##############################################
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

def visualize_and_single_case_demo(model, features, edge_index, edge_weight, labels, method="tsne"):
    model.eval()
    with torch.no_grad():
        # 1. 得到 embedding 和所有得分
        mlp_emb, gcn_emb = model.forward_full(features, edge_index, edge_weight)
        final_score, local_incons, proto_score, neigh_mean, residual = model.final_scores(
            mlp_emb, gcn_emb, edge_index, alpha_proto=args.alpha_proto
        )
        halo = compute_HALO_emb(mlp_emb)

        emb_np = mlp_emb.cpu().numpy()
        scores_np = final_score.cpu().numpy()

        # 2. 降维（用于 Origin 画图）
        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
        else:
            reducer = PCA(n_components=2)

        coords = reducer.fit_transform(emb_np)
        df = pd.DataFrame({
            "node": np.arange(len(coords)),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "final_score": scores_np,
            "halo": halo.cpu().numpy(),
            "local_incons": local_incons.cpu().numpy(),
            "proto_score": proto_score.cpu().numpy(),
            "label": labels.cpu().numpy()
        })

        # 3. 导出用于 Origin 的 CSV
        df.to_csv("node_scores_for_origin.csv", index=False)
        print("已导出可视化 CSV：node_scores_for_origin.csv")

        # 4. 选一个正常节点、一个异常节点展示评分
        normal_nodes = (labels == 0).nonzero().squeeze()
        anom_nodes = (labels == 1).nonzero().squeeze()

        normal_id = normal_nodes[torch.randint(0, len(normal_nodes), (1,))].item()
        anom_id = anom_nodes[torch.randint(0, len(anom_nodes), (1,))].item()

        def print_node_info(idx, title):
            print("\n===============================")
            print(f"{title} 节点 ID = {idx}")
            print("===============================")
            print("HALO:", float(halo[idx]))
            print("Local inconsistency:", float(local_incons[idx]))
            print("Prototype score:", float(proto_score[idx]))
            print("Final anomaly score:", float(final_score[idx]))
            print("Residual vector:", residual[idx].cpu().numpy())

        print_node_info(normal_id, "正常节点 Normal")
        print_node_info(anom_id, "异常节点 Abnormal")

        return df

def find_prototype_anomaly(model, features, edge_index, edge_weight, labels, top_k=1):
    model.eval()
    with torch.no_grad():
        mlp_emb, gcn_emb = model.forward_full(features, edge_index, edge_weight)
        final_score, local_incons, proto_score, neigh_mean, residual = model.final_scores(
            mlp_emb, gcn_emb, edge_index, alpha_proto=args.alpha_proto
        )

        # # 只考虑真实异常节点
        # anomaly_idx = (labels == 1).nonzero().squeeze()
        #
        # fi = final_score[anomaly_idx]
        # lo = local_incons[anomaly_idx]
        # pr = proto_score[anomaly_idx]
        #
        # # 目标：Proto 高，但 Local 低的节点
        # # 排序依据 = proto_score - local_incons （越高越偏向依赖原型）
        # score_balance = pr - lo
        # _, idx_sorted = torch.sort(score_balance, descending=True)
        #
        # top_nodes = anomaly_idx[idx_sorted[:top_k]]
        #
        # results = []
        # for nid in top_nodes:
        #     results.append({
        #         "node_id": int(nid),
        #         "final_score": float(final_score[nid]),
        #         "proto_score": float(proto_score[nid]),
        #         "local_incons": float(local_incons[nid]),
        #         "residual": residual[nid].cpu().numpy()
        #     })
        #
        # return results
        anomaly_idx = (labels == 1).nonzero().squeeze()

        fi = final_score[anomaly_idx]
        lo = local_incons[anomaly_idx]
        pr = proto_score[anomaly_idx]

        # 关键：Proto 高且 Local 低 → Proto 主导型异常
        proto_dom_metric = pr - lo

        # 取该 metric 最大的一个节点
        best_idx = torch.argmax(proto_dom_metric)
        node_id = int(anomaly_idx[best_idx])

        print("\n====== 原型主导型异常实例（Prototype-Dominant Case）======")
        print(f"节点 ID: {node_id}")
        print(f"Final Score: {float(final_score[node_id]):.4f}")
        print(f"Proto Score: {float(proto_score[node_id]):.4f}")
        print(f"Local Incons: {float(local_incons[node_id]):.4f} ")
        print(f"Residual Vector:\n{residual[node_id].cpu().numpy()}")

        return node_id


##############################################
# Case study: find TWO anomaly nodes in the SAME 1-hop neighborhood
#  - one is "local-detected" (high local_incons)
#  - the other is "global-detected" (high proto_score but low local_incons)
# Also export the whole 1-hop neighborhood for plotting.
##############################################

def _build_adj_list(edge_index, num_nodes: int):
    """Build 1-hop neighbor lists from edge_index (assumed undirected)."""
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[int(s)].append(int(d))
    # de-dup while keeping stable-ish order
    for i in range(num_nodes):
        if len(adj[i]) <= 1:
            continue
        adj[i] = list(dict.fromkeys(adj[i]))
    return adj


def _inverse_rank_desc(scores_1d: torch.Tensor):
    """Return 1-based rank where rank=1 means the largest score."""
    order = torch.argsort(scores_1d, descending=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=scores_1d.device)
    return (inv + 1)


def find_case_in_same_neighborhood(model, features, edge_index, edge_weight, labels,
                                   local_q=0.70, proto_q=0.70, local_low_q=0.40, diff_q=0.70,
                                   max_trials=4, seed=42):
    """Find a 1-hop ego neighborhood that contains:
    - an anomaly with high local_incons (local-detected)
    - a different anomaly with high proto_score but low local_incons (global-detected)
    Returns (center_id, local_anom_id, global_anom_id, ego_nodes, pack_dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    with torch.no_grad():
        mlp_emb, gcn_emb = model.forward_full(features, edge_index, edge_weight)
        final_score, local_incons, proto_score, _, _ = model.final_scores(
            mlp_emb, gcn_emb, edge_index, alpha_proto=args.alpha_proto
        )
        halo = compute_HALO_emb(mlp_emb, None, epsilon=1e-8, chunk_size=args.halo_chunk)

    y = labels.detach().cpu()
    N = y.numel()
    adj = _build_adj_list(edge_index, N)

    # Only consider true anomalies as candidates
    anom_idx = (y == 1).nonzero(as_tuple=False).view(-1)
    if anom_idx.numel() < 2:
        raise RuntimeError("Not enough anomaly nodes to build a case study.")

    # Work on CPU tensors for quantiles
    li = local_incons.detach().cpu()[anom_idx]
    ps = proto_score.detach().cpu()[anom_idx]
    df = (ps - li)

    def qval(t: torch.Tensor, q: float) -> float:
        return float(torch.quantile(t, torch.tensor(q)))

    # progressive relaxation if not found
    cfgs = [
        (local_q, proto_q, local_low_q, diff_q),
        (max(0.60, local_q - 0.10), max(0.60, proto_q - 0.10), min(0.50, local_low_q + 0.05), max(0.60, diff_q - 0.10)),
        (max(0.55, local_q - 0.15), max(0.55, proto_q - 0.15), min(0.55, local_low_q + 0.10), max(0.55, diff_q - 0.15)),
    ]

    # Precompute global HALO ranks (1 = highest)
    halo_cpu = halo.detach().cpu()
    halo_rank_global = _inverse_rank_desc(halo_cpu)

    for (lq, pq, llq, dq) in cfgs[:max_trials]:
        li_hi = qval(li, lq)
        li_lo = qval(li, llq)
        ps_hi = qval(ps, pq)
        df_hi = qval(df, dq)

        local_candidates = anom_idx[(li >= li_hi)]
        global_candidates = anom_idx[(ps >= ps_hi) & (li <= li_lo) & ((ps - li) >= df_hi)]

        if local_candidates.numel() == 0 or global_candidates.numel() == 0:
            continue

        # Shuffle local candidates to avoid always picking the same node
        perm = torch.randperm(local_candidates.numel())
        local_candidates = local_candidates[perm]

        global_set = set(global_candidates.tolist())

        for a_local in local_candidates.tolist():
            neigh = adj[int(a_local)]
            ego = [int(a_local)] + neigh
            ego_set = set(ego)

            # Find a global-detected anomaly inside the same 1-hop neighborhood
            g_in_ego = [n for n in ego if (n in global_set) and (n != int(a_local))]
            if len(g_in_ego) == 0:
                continue
            a_global = int(g_in_ego[0])

            # pick a normal node in this ego for contrast (if any)
            normals = [n for n in ego if y[n].item() == 0]
            normal_pick = int(normals[0]) if len(normals) > 0 else None

            # neighborhood HALO rank (within ego)
            ego_tensor = torch.tensor(ego, dtype=torch.long)
            halo_ego = halo_cpu[ego_tensor]
            halo_rank_ego = _inverse_rank_desc(halo_ego)

            pack = {
                "final_score": final_score.detach().cpu(),
                "local_incons": local_incons.detach().cpu(),
                "proto_score": proto_score.detach().cpu(),
                "halo": halo_cpu,
                "halo_rank_global": halo_rank_global,
                "halo_rank_ego": (ego_tensor, halo_rank_ego),
                "normal_pick": normal_pick,
                "thresholds": {"li_hi": li_hi, "li_lo": li_lo, "ps_hi": ps_hi, "df_hi": df_hi,
                               "lq": lq, "pq": pq, "llq": llq, "dq": dq},
            }
            return int(a_local), int(a_local), a_global, ego, pack

    raise RuntimeError(
        "Failed to find two anomaly nodes in the same 1-hop neighborhood with the desired properties. "
        "Try relaxing quantiles or using a different random seed."
    )


def export_case_neighborhood_csv(csv_path: str, center_id: int, local_anom_id: int, global_anom_id: int,
                                 ego_nodes: list, labels, pack: dict):
    """Export a CSV with all nodes in the ego neighborhood and their scores for plotting."""
    y = labels.detach().cpu().numpy()
    final_score = pack["final_score"].numpy()
    local_incons = pack["local_incons"].numpy()
    proto_score = pack["proto_score"].numpy()
    halo = pack["halo"].numpy()
    halo_rank_global = pack["halo_rank_global"].numpy()

    ego_tensor, halo_rank_ego = pack["halo_rank_ego"]
    halo_rank_ego = halo_rank_ego.detach().cpu().numpy()
    ego_tensor = ego_tensor.detach().cpu().numpy()
    halo_rank_in_ego = {int(n): int(r) for n, r in zip(ego_tensor, halo_rank_ego)}

    rows = []
    for n in ego_nodes:
        role = "other"
        if n == local_anom_id:
            role = "anom_local"
        elif n == global_anom_id:
            role = "anom_global"
        elif y[n] == 0:
            role = "normal"
        rows.append({
            "node": int(n),
            "is_center": int(n == center_id),
            "role": role,
            "label": int(y[n]),
            "local_incons": float(local_incons[n]),
            "global_proto": float(proto_score[n]),
            "final_score": float(final_score[n]),
            "halo": float(halo[n]),
            "halo_rank_global": int(halo_rank_global[n]),
            "halo_rank_in_ego": int(halo_rank_in_ego.get(int(n), -1)),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"已导出 case study 邻域 CSV：{csv_path}")


def print_case_neighborhood(center_id: int, local_anom_id: int, global_anom_id: int, ego_nodes: list,
                            labels, pack: dict):
    """Pretty print: neighborhood nodes + scores + HALO ranks."""
    y = labels.detach().cpu().numpy()
    final_score = pack["final_score"].numpy()
    local_incons = pack["local_incons"].numpy()
    proto_score = pack["proto_score"].numpy()
    halo = pack["halo"].numpy()
    halo_rank_global = pack["halo_rank_global"].numpy()

    ego_tensor, halo_rank_ego = pack["halo_rank_ego"]
    halo_rank_ego = halo_rank_ego.detach().cpu().numpy()
    ego_tensor = ego_tensor.detach().cpu().numpy()
    halo_rank_in_ego = {int(n): int(r) for n, r in zip(ego_tensor, halo_rank_ego)}

    th = pack.get("thresholds", {})
    print("\n================= Case Study (Same 1-hop Neighborhood) =================")
    print(f"Center (ego) node: {center_id} | ego_size={len(ego_nodes)}")
    print(f"Anomaly (local-detected): {local_anom_id} | label=1")
    print(f"Anomaly (global-detected): {global_anom_id} | label=1")
    if pack.get("normal_pick") is not None:
        print(f"Example normal node in ego: {pack['normal_pick']} | label=0")
    if th:
        print("Thresholds (quantile-based):", th)

    print("\n[HALO ranking within ego] (rank=1 means highest HALO in this ego)")
    ego_sorted = sorted(ego_nodes, key=lambda n: halo_rank_in_ego.get(int(n), 10**9))
    print(" ".join([f"{n}(r{halo_rank_in_ego.get(int(n), -1)})" for n in ego_sorted]))

    print("\n[All nodes in ego: scores & HALO ranks]")
    header = f"{'node':>7} | {'role':>10} | {'y':>1} | {'local':>8} | {'global':>8} | {'final':>8} | {'HALO':>8} | {'R_g':>4} | {'R_ego':>5}"
    print(header)
    print("-" * len(header))
    for n in ego_sorted:
        role = "other"
        if int(n) == local_anom_id:
            role = "anom_local"
        elif int(n) == global_anom_id:
            role = "anom_global"
        elif y[int(n)] == 0:
            role = "normal"
        print(f"{int(n):7d} | {role:>10} | {int(y[int(n)]):1d} | {local_incons[int(n)]:8.4f} | {proto_score[int(n)]:8.4f} | {final_score[int(n)]:8.4f} | {halo[int(n)]:8.4f} | {int(halo_rank_global[int(n)]):4d} | {int(halo_rank_in_ego.get(int(n), -1)):5d}")



# -----------------------
# main
# -----------------------
def main():
    features, edge_index, edge_weight, labels = load_data(args.data_dir, args.dataset)
    train_mask, val_mask, test_mask = split_dataset(labels, train_ratio=0.7, val_ratio=0.1, seed=args.seed)
    model = ModelB(in_dim=features.size(1), hidden_dim=args.hidden_dim, out_dim=args.output_dim, gcn_layers=args.gcn_layers, dropout=args.dropout).to(device)
    print("Trainable params:")
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)
    trained_model, best_val, best_epoch = train(model, features, edge_index, edge_weight, labels, train_mask, val_mask)
    print(f"Training finished. Best val auc={best_val:.4f} at epoch {best_epoch}")
    test_auc, test_ap = evaluate(trained_model, features, edge_index, edge_weight, labels, test_mask)
    print(f"Final Test - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

    # -----------------------
    # Case study: two anomalies in the SAME 1-hop neighborhood
    # -----------------------
    try:
        center_id, local_anom_id, global_anom_id, ego_nodes, pack = find_case_in_same_neighborhood(
            trained_model, features, edge_index, edge_weight, labels,
            seed=args.seed
        )
        print_case_neighborhood(center_id, local_anom_id, global_anom_id, ego_nodes, labels, pack)
        export_case_neighborhood_csv(
            csv_path="case_study_ego_nodes.csv",
            center_id=center_id,
            local_anom_id=local_anom_id,
            global_anom_id=global_anom_id,
            ego_nodes=ego_nodes,
            labels=labels,
            pack=pack,
        )
    except RuntimeError as e:
        print("[CaseStudy]", str(e))

if __name__ == "__main__":
    main()





