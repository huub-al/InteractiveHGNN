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

def train_and_measure_time(model, data, epochs, lr, device='cpu', weight_decay=5e-4):
    """
    Train the model and measure training time.
    
    Args:
        model: The model to train
        data: The subgraph data containing embeddings, incidence matrix, labels, and masks
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        weight_decay: Weight decay for optimizer
        
    Returns:
        float: Training time in seconds
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.process_time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.embeddings.to(device), data.incidence.to(device))
        loss = criterion(out[data.train_mask], data.labels[data.train_mask].to(device))
        loss.backward()
        optimizer.step()

    end_time = time.process_time()
    training_time = end_time - start_time
    
    return training_time

def train_interactive_model_and_measure_time(baseline_model, neighborhoods, epochs_per_neighborhood, lr, device='cpu'):
    """
    Train an interactive model by sequentially training on each neighborhood and measure total time.
    
    Args:
        baseline_model: The baseline model to start from
        neighborhoods: List of neighborhood subgraphs
        epochs_per_neighborhood: Number of epochs to train on each neighborhood
        lr: Learning rate
        device: Device to train on
        
    Returns:
        float: Total training time in seconds
    """
    interactive_model = copy.deepcopy(baseline_model)
    
    start_time = time.process_time()
    
    for i, neighborhood in enumerate(neighborhoods):
        print(f"  Training on neighborhood {i+1}/{len(neighborhoods)} for {epochs_per_neighborhood} epochs...")
        
        # Train on this neighborhood
        optimizer = torch.optim.Adam(interactive_model.parameters(), lr=lr, weight_decay=0)
        criterion = torch.nn.CrossEntropyLoss()
        
        interactive_model.train()
        for epoch in range(epochs_per_neighborhood):
            optimizer.zero_grad()
            out = interactive_model(neighborhood.embeddings.to(device), neighborhood.incidence.to(device))
            loss = criterion(out[neighborhood.train_mask], neighborhood.labels[neighborhood.train_mask].to(device))
            loss.backward()
            optimizer.step()
    
    end_time = time.process_time()
    total_training_time = end_time - start_time
    
    return total_training_time

def run_timing_experiment(device='cpu'):
    print("\n===== Running Training Time Comparison Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    subgraph = hypergraph.construct_subgraph(dropout=0.2)
    
    # Get initial graph size
    num_nodes_full = subgraph.embeddings.shape[0]
    print(f"Full graph size: {num_nodes_full} nodes")

    # Initialize model parameters
    in_dim = subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)

    # Step 2: Train master model on full graph
    print("\nTraining master model on full graph...")
    master_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    master_time = train_and_measure_time(master_model, subgraph, epochs=100, lr=1e-3, device=device, weight_decay=0)
    print(f"Master model training time: {master_time:.2f} seconds")

    # Step 3: Remove outliers and train baseline model
    print("\nRemoving outliers and training baseline model...")
    subgraph.remove_outliers(outlier_fraction=0.01)
    num_nodes_no_outliers = subgraph.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")
    print(f"Removed {num_nodes_full - num_nodes_no_outliers} outliers")
    
    # Train baseline model (this is needed for the interactive approach)
    baseline_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    baseline_time = train_and_measure_time(baseline_model, subgraph, epochs=100, lr=1e-3, device=device, weight_decay=0)
    print(f"Baseline model training time: {baseline_time:.2f} seconds")

    # Step 4: Create large neighborhood graph (for naive approach)
    print("\nConstructing large neighborhood graph...")
    large_neighborhood_all_masks = subgraph.construct_outlier_neighbourhood(False)
    if large_neighborhood_all_masks is None:
        print("No outliers found to construct neighborhood graph!")
        return None
    
    num_nodes_large_neighborhood = large_neighborhood_all_masks.embeddings.shape[0]
    print(f"Large neighborhood graph size: {num_nodes_large_neighborhood} nodes")
    print(f"Large neighborhood is {num_nodes_large_neighborhood/num_nodes_full*100:.1f}% the size of the full graph")

    # Step 5: Train naive model on large neighborhood
    print("\nTraining naive model on large neighborhood...")
    naive_model = copy.deepcopy(baseline_model)
    naive_time = train_and_measure_time(naive_model, large_neighborhood_all_masks, epochs=100, lr=1e-4, device=device, weight_decay=0)
    print(f"Naive model training time: {naive_time:.2f} seconds")

    # Step 6: Create small neighborhoods (for interactive approach)
    print("\nGenerating small neighborhoods...")
    outliers_per_subgraph = 25
    small_neighborhoods = subgraph.generate_neighbourhoods(outliers_per_subgraph=outliers_per_subgraph)
    print(f"Generated {len(small_neighborhoods)} small neighborhoods")
    
    total_nodes_small_neighborhoods = sum(len(neighborhood.sub_idx_to_main) for neighborhood in small_neighborhoods)
    print(f"Total nodes across all small neighborhoods: {total_nodes_small_neighborhoods}")
    
    for i, neighborhood in enumerate(small_neighborhoods):
        print(f"  Neighborhood {i}: {len(neighborhood.sub_idx_to_main)} nodes")

    # Step 7: Train interactive model on small neighborhoods
    print("\nTraining interactive model on small neighborhoods...")
    interactive_time = train_interactive_model_and_measure_time(
        baseline_model=baseline_model, 
        neighborhoods=small_neighborhoods, 
        epochs_per_neighborhood=100, 
        lr=1e-4, 
        device=device
    )
    print(f"Interactive model training time: {interactive_time:.2f} seconds")

    # Calculate speedups
    master_to_naive_speedup = master_time / naive_time
    master_to_interactive_speedup = master_time / interactive_time
    naive_to_interactive_speedup = naive_time / interactive_time

    print(f"\nSpeedup Analysis:")
    print(f"Master vs Naive: {master_to_naive_speedup:.2f}x faster")
    print(f"Master vs Interactive: {master_to_interactive_speedup:.2f}x faster")
    print(f"Naive vs Interactive: {naive_to_interactive_speedup:.2f}x faster")

    return {
        'master': {
            'time': master_time,
            'nodes': num_nodes_full,
            'description': 'Full Graph'
        },
        'naive': {
            'time': naive_time,
            'nodes': num_nodes_large_neighborhood,
            'description': 'Large Neighborhood'
        },
        'interactive': {
            'time': interactive_time,
            'nodes': total_nodes_small_neighborhoods,
            'num_neighborhoods': len(small_neighborhoods),
            'description': 'Multiple Small Neighborhoods'
        },
        'baseline': {
            'time': baseline_time,
            'nodes': num_nodes_no_outliers,
            'description': 'Baseline (No Outliers)'
        }
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run timing experiment
    timing_results = run_timing_experiment(device=device)
    if timing_results is None:
        return

    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    approaches = ['Master', 'Naive', 'Interactive']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Training Time plot
    times = [timing_results['master']['time'], timing_results['naive']['time'], timing_results['interactive']['time']]
    bars1 = ax1.bar(approaches, times, color=colors, alpha=0.8)
    ax1.set_ylabel('CPU Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels for training time
    for bar, time in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2., time,
                f'{time:.1f}s', ha='center', va='bottom')

    # Graph Size plot
    sizes = [timing_results['master']['nodes'], timing_results['naive']['nodes'], timing_results['interactive']['nodes']]
    bars2 = ax2.bar(approaches, sizes, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Nodes')
    ax2.set_title('Graph Size Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels for graph sizes
    for bar, size in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2., size,
                f'{size:,}', ha='center', va='bottom')

    # Speedup vs Master plot
    master_time = timing_results['master']['time']
    speedups = [1.0, master_time / timing_results['naive']['time'], master_time / timing_results['interactive']['time']]
    bars3 = ax3.bar(approaches, speedups, color=colors, alpha=0.8)
    ax3.set_ylabel('Speedup Factor (vs Master)')
    ax3.set_title('Training Speedup vs Master Model')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels for speedups
    for bar, speedup in zip(bars3, speedups):
        ax3.text(bar.get_x() + bar.get_width()/2., speedup,
                f'{speedup:.1f}x', ha='center', va='bottom')

    # Time per Node plot
    time_per_node = [timing_results['master']['time'] / timing_results['master']['nodes'],
                    timing_results['naive']['time'] / timing_results['naive']['nodes'],
                    timing_results['interactive']['time'] / timing_results['interactive']['nodes']]
    bars4 = ax4.bar(approaches, time_per_node, color=colors, alpha=0.8)
    ax4.set_ylabel('Time per Node (seconds)')
    ax4.set_title('Training Efficiency (Time per Node)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels for time per node
    for bar, tpn in zip(bars4, time_per_node):
        ax4.text(bar.get_x() + bar.get_width()/2., tpn,
                f'{tpn:.4f}s', ha='center', va='bottom')

    plt.suptitle('Training Time Comparison: Master vs Naive vs Interactive Approaches\n'
                f'Master: Full Graph | Naive: Large Neighborhood | Interactive: {timing_results["interactive"]["num_neighborhoods"]} Small Neighborhoods')
    plt.tight_layout()
    plt.savefig("training_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed summary
    print("\n" + "="*60)
    print("DETAILED TIMING SUMMARY")
    print("="*60)
    print(f"Master Model (Full Graph):")
    print(f"  - Training Time: {timing_results['master']['time']:.2f} seconds")
    print(f"  - Graph Size: {timing_results['master']['nodes']:,} nodes")
    print(f"  - Time per Node: {timing_results['master']['time']/timing_results['master']['nodes']:.6f} seconds")
    
    print(f"\nNaive Model (Large Neighborhood):")
    print(f"  - Training Time: {timing_results['naive']['time']:.2f} seconds")
    print(f"  - Graph Size: {timing_results['naive']['nodes']:,} nodes")
    print(f"  - Time per Node: {timing_results['naive']['time']/timing_results['naive']['nodes']:.6f} seconds")
    print(f"  - Speedup vs Master: {master_time/timing_results['naive']['time']:.2f}x")
    
    print(f"\nInteractive Model (Multiple Small Neighborhoods):")
    print(f"  - Training Time: {timing_results['interactive']['time']:.2f} seconds")
    print(f"  - Total Graph Size: {timing_results['interactive']['nodes']:,} nodes")
    print(f"  - Time per Node: {timing_results['interactive']['time']/timing_results['interactive']['nodes']:.6f} seconds")
    print(f"  - Speedup vs Master: {master_time/timing_results['interactive']['time']:.2f}x")
    print(f"  - Speedup vs Naive: {timing_results['naive']['time']/timing_results['interactive']['time']:.2f}x")

if __name__ == "__main__":
    main() 