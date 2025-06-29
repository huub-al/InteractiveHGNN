import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/Users/huubal/scriptie/data")
sys.path.append("/Users/huubal/scriptie/paperNodes")
from paperNodes_graph import arXivHyperGraph
from model import arXivHGNN

def train(model, data, epochs, lr, device='cpu', weight_decay=5e-4):
    """
    Train the model using the built-in train and validation masks.
    
    Args:
        model: The model to train
        data: The subgraph data containing embeddings, incidence matrix, labels, and masks
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.embeddings.to(device), data.incidence.to(device))
        loss = criterion(out[data.train_mask], data.labels[data.train_mask].to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.embeddings.to(device), data.incidence.to(device))
        pred = out.argmax(dim=1)
        correct = pred[data.val_mask].eq(data.labels[data.val_mask].to(device)).sum().item()
        acc = correct / data.val_mask.sum().item()
    return acc

def evaluate_model(model, data, device='cpu'):
    """
    Evaluate a model on a specific validation set.
    
    Args:
        model: The model to evaluate
        data: The subgraph data containing embeddings, incidence matrix, and labels
        device: Device to evaluate on
    """
    model.eval()
    with torch.no_grad():
        out = model(data.embeddings.to(device), data.incidence.to(device))
        pred = out.argmax(dim=1)
        correct = pred[data.val_mask].eq(data.labels[data.val_mask].to(device)).sum().item()
        acc = correct / data.val_mask.sum().to(device).item()
    return acc

def train_interactive_model(baseline_model, neighborhoods, epochs_per_neighborhood, lr, device='cpu', master_subgraph=None, large_neighborhood=None):
    """
    Train an interactive model by sequentially training on each neighborhood.
    
    Args:
        baseline_model: The baseline model to start from
        neighborhoods: List of neighborhood subgraphs
        epochs_per_neighborhood: Number of epochs to train on each neighborhood
        lr: Learning rate
        device: Device to train on
        master_subgraph: The master subgraph for evaluation
        large_neighborhood: The large neighborhood for evaluation
    
    Returns:
        The trained interactive model
    """
    interactive_model = copy.deepcopy(baseline_model)
    
    for i, neighborhood in enumerate(neighborhoods):
        print(f"Training on neighborhood {i+1}/{len(neighborhoods)} for {epochs_per_neighborhood} epochs...")
        
        # Train on this neighborhood
        train(interactive_model, neighborhood, epochs=epochs_per_neighborhood, lr=lr, device=device)
        
        # Evaluate current performance if evaluation data is provided
        if master_subgraph is not None and large_neighborhood is not None:
            master_acc = evaluate_model(interactive_model, master_subgraph, device)
            neighborhood_acc = evaluate_model(interactive_model, large_neighborhood, device)
            print(f"  After neighborhood {i+1}: Master acc: {master_acc:.4f}, Neighborhood acc: {neighborhood_acc:.4f}")
    
    return interactive_model

def run_incremental_experiment(device='cpu'):
    print("\n===== Running Incremental Training Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    subgraph = hypergraph.construct_subgraph(dropout=0.2)
    
    # Initialize model parameters
    in_dim = subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)

    # Step 2: Train master model on full graph
    print("\nTraining master model on full graph...")
    master_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    master_acc = train(master_model, subgraph, epochs=100, lr=1e-3, device=device, weight_decay=0)
    print(f"Master model accuracy: {master_acc:.4f}")

    # Step 3: Remove outliers and train baseline model
    print("\nRemoving outliers and training baseline model...")
    subgraph.remove_outliers(outlier_fraction=0.01)
    num_nodes_no_outliers = subgraph.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")
    
    # Train baseline model using the automatically generated masks
    baseline_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    baseline_acc = train(baseline_model, subgraph, epochs=100, lr=1e-3, device=device, weight_decay=0)
    print(f"Baseline model accuracy: {baseline_acc:.4f}")

    # Step 4: Create large neighborhood graph
    print("\nConstructing large neighborhood graph...")
    large_neighborhood = subgraph.construct_outlier_neighbourhood(True)
    large_neighborhood_all_masks = subgraph.construct_outlier_neighbourhood(False)
    if large_neighborhood is None:
        print("No outliers found to construct neighborhood graph!")
        return None
    
    num_nodes_large_neighborhood = large_neighborhood.embeddings.shape[0]
    print(f"Large neighborhood graph size: {num_nodes_large_neighborhood} nodes")

    # Step 5: Create small neighborhoods
    print("\nGenerating small neighborhoods...")
    outliers_per_subgraph = 25
    small_neighborhoods = subgraph.generate_neighbourhoods(outliers_per_subgraph=outliers_per_subgraph)
    print(f"Generated {len(small_neighborhoods)} small neighborhoods")
    
    for i, neighborhood in enumerate(small_neighborhoods):
        print(f"  Neighborhood {i}: {len(neighborhood.sub_idx_to_main)} nodes")

    # Step 6: Test different epoch ranges
    epochs_range = range(10, 50)  # 3 to 30 epochs
    learning_rate = 1e-4
    incremental_lr = 1e-4
    
    naive_results = {'master': [], 'neighborhood': []}
    interactive_results = {'master': [], 'neighborhood': []}
    
    print(f"\nTesting epochs range: {list(epochs_range)}")
    print(f"Learning rate: {learning_rate}")
    
    for epochs in epochs_range:
        print(f"\n--- Testing {epochs} epochs ---")
        
        # Train naive model on large neighborhood (without outlier masks)
        print("Training naive model on large neighborhood (without outlier masks)...")
        naive_model = copy.deepcopy(baseline_model)
        train(naive_model, large_neighborhood_all_masks, epochs=epochs, lr=learning_rate, device=device)
        
        # Evaluate naive model on master graph and neighborhood with outlier masks
        naive_master_acc = evaluate_model(naive_model, subgraph, device)
        naive_neighborhood_acc = evaluate_model(naive_model, large_neighborhood, device)
        naive_results['master'].append(naive_master_acc)
        naive_results['neighborhood'].append(naive_neighborhood_acc)
        
        print(f"Naive model - Master acc: {naive_master_acc:.4f}, Neighborhood acc: {naive_neighborhood_acc:.4f}")
        
        # Train interactive model on small neighborhoods
        print("Training interactive model on small neighborhoods...")
        interactive_model = train_interactive_model(
            baseline_model, 
            small_neighborhoods, 
            epochs, 
            incremental_lr, 
            device,
            subgraph,
            large_neighborhood
        )
        
        # Evaluate interactive model
        interactive_master_acc = evaluate_model(interactive_model, subgraph, device)
        interactive_neighborhood_acc = evaluate_model(interactive_model, large_neighborhood, device)
        interactive_results['master'].append(interactive_master_acc)
        interactive_results['neighborhood'].append(interactive_neighborhood_acc)
        
        print(f"Interactive model - Master acc: {interactive_master_acc:.4f}, Neighborhood acc: {interactive_neighborhood_acc:.4f}")

    # Step 7: Evaluate baseline and master models
    print("\nEvaluating baseline and master models...")
    
    baseline_master_acc = evaluate_model(baseline_model, subgraph, device)
    baseline_neighborhood_acc = evaluate_model(baseline_model, large_neighborhood, device)
    master_master_acc = evaluate_model(master_model, subgraph, device)
    master_neighborhood_acc = evaluate_model(master_model, large_neighborhood, device)
    
    print(f"Baseline model - Master acc: {baseline_master_acc:.4f}, Neighborhood acc: {baseline_neighborhood_acc:.4f}")
    print(f"Master model - Master acc: {master_master_acc:.4f}, Neighborhood acc: {master_neighborhood_acc:.4f}")
    
    return {
        'naive_results': naive_results,
        'interactive_results': interactive_results,
        'baseline_results': {
            'master_acc': baseline_master_acc,
            'neighborhood_acc': baseline_neighborhood_acc
        },
        'master_results': {
            'master_acc': master_master_acc,
            'neighborhood_acc': master_neighborhood_acc
        },
        'epochs_range': list(epochs_range),
        'num_neighborhoods': len(small_neighborhoods)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_repeats = 15 # Reduced for faster execution

    # Initialize results containers
    all_naive_results = {'master': [], 'neighborhood': []}
    all_interactive_results = {'master': [], 'neighborhood': []}
    all_baseline_results = {'master_acc': [], 'neighborhood_acc': []}
    all_master_results = {'master_acc': [], 'neighborhood_acc': []}

    # Run experiments
    for repeat in range(num_repeats):
        print(f"\n===== REPEAT {repeat + 1}/{num_repeats} =====")
        results = run_incremental_experiment(device=device)
        if results is None:
            continue

        # Store results
        all_naive_results['master'].append(results['naive_results']['master'])
        all_naive_results['neighborhood'].append(results['naive_results']['neighborhood'])
        all_interactive_results['master'].append(results['interactive_results']['master'])
        all_interactive_results['neighborhood'].append(results['interactive_results']['neighborhood'])
        all_baseline_results['master_acc'].append(results['baseline_results']['master_acc'])
        all_baseline_results['neighborhood_acc'].append(results['baseline_results']['neighborhood_acc'])
        all_master_results['master_acc'].append(results['master_results']['master_acc'])
        all_master_results['neighborhood_acc'].append(results['master_results']['neighborhood_acc'])

    # Compute averages
    avg_naive_results = {
        'master': np.mean(all_naive_results['master'], axis=0),
        'neighborhood': np.mean(all_naive_results['neighborhood'], axis=0)
    }
    avg_interactive_results = {
        'master': np.mean(all_interactive_results['master'], axis=0),
        'neighborhood': np.mean(all_interactive_results['neighborhood'], axis=0)
    }
    avg_baseline_results = {
        'master_acc': np.mean(all_baseline_results['master_acc']),
        'neighborhood_acc': np.mean(all_baseline_results['neighborhood_acc'])
    }
    avg_master_results = {
        'master_acc': np.mean(all_master_results['master_acc']),
        'neighborhood_acc': np.mean(all_master_results['neighborhood_acc'])
    }

    # Print average results
    print("\n" + "="*50)
    print("AVERAGE RESULTS")
    print("="*50)
    print(f"\nBaseline Model:")
    print(f"Master Graph Accuracy: {avg_baseline_results['master_acc']:.4f}")
    print(f"Neighborhood Graph Accuracy: {avg_baseline_results['neighborhood_acc']:.4f}")
    print(f"\nMaster Model:")
    print(f"Master Graph Accuracy: {avg_master_results['master_acc']:.4f}")
    print(f"Neighborhood Graph Accuracy: {avg_master_results['neighborhood_acc']:.4f}")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = results['epochs_range']
    
    # Plot 1: Master Graph Validation Accuracy
    ax1.plot(epochs, avg_naive_results['master'], marker='o', label='Naive (Large Neighborhood, no outlier masks)', linewidth=2)
    ax1.plot(epochs, avg_interactive_results['master'], marker='s', label='Interactive (Small Neighborhoods)', linewidth=2)
    
    # Plot baseline and master model results as horizontal lines
    ax1.axhline(y=avg_baseline_results['master_acc'], 
                color='r', linestyle='--', 
                label=f'Baseline (100 epochs)')
    ax1.axhline(y=avg_master_results['master_acc'], 
                color='g', linestyle='--', 
                label=f'Master (100 epochs)')
    
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Master Graph Validation Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Neighborhood Graph Validation Accuracy
    ax2.plot(epochs, avg_naive_results['neighborhood'], marker='o', label='Naive (Large Neighborhood, no outlier masks)', linewidth=2)
    ax2.plot(epochs, avg_interactive_results['neighborhood'], marker='s', label='Interactive (Small Neighborhoods)', linewidth=2)
    
    # Plot baseline and master model results as horizontal lines
    ax2.axhline(y=avg_baseline_results['neighborhood_acc'], 
                color='r', linestyle='--', 
                label=f'Baseline (100 epochs)')
    ax2.axhline(y=avg_master_results['neighborhood_acc'], 
                color='g', linestyle='--', 
                label=f'Master (100 epochs)')
    
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Neighborhood Graph Validation Accuracy (Outlier Masks)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f'Incremental Training Comparison: Naive vs Interactive (Averaged over {num_repeats} runs)\n'
                f'Naive: Large Neighborhood (no outlier masks) vs Interactive: {results["num_neighborhoods"]} Small Neighborhoods')
    plt.tight_layout()
    plt.savefig("incremental_training_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 