import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
sys.path.append("/Users/huubal/scriptie/data")
sys.path.append("/Users/huubal/scriptie/paperNodes")
from paperNodes_graph import arXivHyperGraph
from model import arXivHGNN

def train_and_measure_time(model, data, train_mask, val_mask, epochs, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.process_time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.embeddings.to(device), data.incidence.to(device))
        loss = criterion(out[train_mask], data.labels[train_mask].to(device))
        loss.backward()
        optimizer.step()

    end_time = time.process_time()
    training_time = end_time - start_time
    
    return training_time

def get_masks(num_nodes, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    split = int((1 - val_ratio) * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:split]] = True
    val_mask[indices[split:]] = True
    return train_mask, val_mask

def run_timing_experiment(device='cpu'):
    print("\n===== Running Timing Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph with dropout and outlier removal
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    subgraph = hypergraph.construct_subgraph(dropout=0.1)
    
    # Store original state before outlier removal
    original_embeddings = subgraph.embeddings.clone()
    original_labels = subgraph.labels.clone()
    original_node_to_authors = copy.deepcopy(subgraph.node_to_authors)
    
    # Remove outliers
    print("Removing outliers from subgraph...")
    subgraph.remove_outliers(outlier_fraction=0.01)
    num_nodes_no_outliers = subgraph.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")

    # Create masks for training without outliers
    train_mask, val_mask = get_masks(num_nodes_no_outliers)
    baseline_train_size = train_mask.sum().item()
    print(f"Baseline training set size: {baseline_train_size} nodes")

    # Initialize model parameters
    in_dim = subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)

    # Step 2: Train baseline model on clean data and measure time
    print("\nTraining baseline model on clean data...")
    baseline_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    baseline_time = train_and_measure_time(baseline_model, subgraph, train_mask, val_mask, 
                                         epochs=100, device=device)
    print(f"Baseline training time: {baseline_time:.2f} seconds")

    # Step 3: Restore outliers
    print("\nRestoring outliers...")
    subgraph.restore_outliers()
    total_nodes_with_outliers = subgraph.embeddings.shape[0]
    num_outliers = total_nodes_with_outliers - num_nodes_no_outliers
    print(f"Total nodes with outliers: {total_nodes_with_outliers}")
    print(f"Number of outliers: {num_outliers}")

    # Step 4: Measure training times for different approaches
    print("\n--- Measuring Training Times ---")
    
    # Approach 1: Expert-only training (only on outliers)
    expert_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    expert_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
    
    # Create mask for expert training (only outliers)
    expert_train_mask = torch.zeros(total_nodes_with_outliers, dtype=torch.bool)
    expert_train_mask[-num_outliers:] = True
    expert_train_size = expert_train_mask.sum().item()
    print(f"Expert training set size: {expert_train_size} nodes")
    
    expert_time = train_and_measure_time(expert_model, subgraph, expert_train_mask, None,
                                       epochs=50, device=device)
    print(f"Expert-only training time: {expert_time:.2f} seconds")
    
    # Approach 2: Full retraining (all data)
    full_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    full_train_mask = torch.ones(total_nodes_with_outliers, dtype=torch.bool)
    full_train_size = full_train_mask.sum().item()
    print(f"Full retraining set size: {full_train_size} nodes")
    
    full_time = train_and_measure_time(full_model, subgraph, full_train_mask, None,
                                     epochs=100, device=device)
    print(f"Full retraining time: {full_time:.2f} seconds")
    
    return {
        'expert_only': expert_time,
        'full_retraining': full_time,
        'train_sizes': {
            'baseline': baseline_train_size,
            'expert': expert_train_size,
            'full': full_train_size
        }
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run timing experiment
    timing_results = run_timing_experiment(device=device)

    # Create figure with two subplots: training time and training set sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    approaches = ['Expert-only Training', 'Full Retraining']
    
    # Training Time plot
    times = [timing_results['expert_only'], timing_results['full_retraining']]
    bars1 = ax1.bar(approaches, times, alpha=0.8)
    ax1.set_ylabel('CPU Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels for training time
    for bar, time in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2., time,
                f'{time:.1f}s', ha='center', va='bottom')

    # Training Set Size plot
    train_sizes = [timing_results['train_sizes']['expert'],
                  timing_results['train_sizes']['full']]
    bars2 = ax2.bar(approaches, train_sizes, alpha=0.8)
    ax2.set_ylabel('Number of Training Nodes')
    ax2.set_title('Training Set Size Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels for training set sizes
    for bar, size in zip(bars2, train_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2., size,
                f'{size:,}', ha='center', va='bottom')

    plt.suptitle('Training Time and Dataset Size Comparison: Expert-only vs Full Retraining')
    plt.tight_layout()
    plt.savefig("training_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 