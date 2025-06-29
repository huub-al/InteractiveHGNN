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
        acc = correct / data.val_mask.sum().item()
    return acc

def run_comparison_experiment(device='cpu'):
    print("\n===== Running Comparison Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    subgraph = hypergraph.construct_subgraph(dropout=0.1)
    
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

    # Step 4: Create neighborhood graph for interactive training
    print("\nConstructing neighborhood graph for interactive training...")
    neighborhood_graph = subgraph.construct_outlier_neighbourhood()
    if neighborhood_graph is None:
        print("No outliers found to construct neighborhood graph!")
        return None
    
    num_nodes_neighborhood = neighborhood_graph.embeddings.shape[0]
    print(f"Neighborhood graph size: {num_nodes_neighborhood} nodes")
    
    # Test different learning rates and epochs for interactive model
    learning_rates = [1e-6, 5e-6, 1e-5, 5e-5]
    epochs_range = range(3, 51)
    interactive_results = {lr: {'neighborhood': [], 'master': []} for lr in learning_rates}
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        for epochs in epochs_range:
            print(f"Training with {epochs} epochs...")
            interactive_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
            interactive_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
            
            # Train on neighborhood graph
            train(interactive_model, neighborhood_graph, epochs=epochs, lr=lr, device=device)
            
            # Evaluate on both validation sets
            neighborhood_acc = evaluate_model(interactive_model, neighborhood_graph, device)
            master_acc = evaluate_model(interactive_model, subgraph, device)
            
            interactive_results[lr]['neighborhood'].append(neighborhood_acc)
            interactive_results[lr]['master'].append(master_acc)
            
            print(f"Interactive model accuracy (neighborhood): {neighborhood_acc:.4f}")
            print(f"Interactive model accuracy (master): {master_acc:.4f}")

    # Step 5: Evaluate baseline and master models
    print("\nEvaluating models on validation sets...")
    
    # Evaluate baseline and master models on both validation sets
    baseline_full_acc = evaluate_model(baseline_model, subgraph, device)
    baseline_neighborhood_acc = evaluate_model(baseline_model, neighborhood_graph, device)
    master_full_acc = evaluate_model(master_model, subgraph, device)
    master_neighborhood_acc = evaluate_model(master_model, neighborhood_graph, device)
    
    # Store results
    master_results = {
        'full_val_acc': master_full_acc,
        'neighborhood_acc': master_neighborhood_acc
    }
    baseline_results = {
        'full_val_acc': baseline_full_acc,
        'neighborhood_acc': baseline_neighborhood_acc
    }
    
    return {
        'interactive_results': interactive_results,
        'baseline_results': baseline_results,
        'master_results': master_results,
        'epochs_range': list(epochs_range)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_repeats = 5

    # Initialize results containers
    all_interactive_results = {lr: {'neighborhood': [], 'master': []} for lr in [1e-6, 5e-6, 1e-5, 5e-5]}
    all_baseline_results = {'full_val_acc': [], 'neighborhood_acc': []}
    all_master_results = {'full_val_acc': [], 'neighborhood_acc': []}

    # Run experiments
    for repeat in range(num_repeats):
        print(f"\n===== REPEAT {repeat + 1}/{num_repeats} =====")
        results = run_comparison_experiment(device=device)
        if results is None:
            continue

        # Store results
        for lr in results['interactive_results']:
            all_interactive_results[lr]['neighborhood'].append(results['interactive_results'][lr]['neighborhood'])
            all_interactive_results[lr]['master'].append(results['interactive_results'][lr]['master'])
        all_baseline_results['full_val_acc'].append(results['baseline_results']['full_val_acc'])
        all_baseline_results['neighborhood_acc'].append(results['baseline_results']['neighborhood_acc'])
        all_master_results['full_val_acc'].append(results['master_results']['full_val_acc'])
        all_master_results['neighborhood_acc'].append(results['master_results']['neighborhood_acc'])

    # Compute averages
    avg_interactive_results = {
        lr: {
            'neighborhood': np.mean(all_interactive_results[lr]['neighborhood'], axis=0),
            'master': np.mean(all_interactive_results[lr]['master'], axis=0)
        }
        for lr in all_interactive_results
    }
    avg_baseline_results = {
        'full_val_acc': np.mean(all_baseline_results['full_val_acc']),
        'neighborhood_acc': np.mean(all_baseline_results['neighborhood_acc'])
    }
    avg_master_results = {
        'full_val_acc': np.mean(all_master_results['full_val_acc']),
        'neighborhood_acc': np.mean(all_master_results['neighborhood_acc'])
    }

    # Print average results
    print("\n" + "="*50)
    print("AVERAGE RESULTS")
    print("="*50)
    print("\nBaseline Model:")
    print(f"Full Graph Validation Accuracy: {avg_baseline_results['full_val_acc']:.4f}")
    print(f"Neighborhood Graph Validation Accuracy: {avg_baseline_results['neighborhood_acc']:.4f}")
    print("\nMaster Model:")
    print(f"Full Graph Validation Accuracy: {avg_master_results['full_val_acc']:.4f}")
    print(f"Neighborhood Graph Validation Accuracy: {avg_master_results['neighborhood_acc']:.4f}")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Master Graph Validation Accuracy
    epochs = results['epochs_range']
    learning_rates = list(avg_interactive_results.keys())
    
    # Plot interactive model results for each learning rate
    for lr in learning_rates:
        accuracies = avg_interactive_results[lr]['master']
        ax1.plot(epochs, accuracies, marker='o', label=f'Interactive (lr={lr})')
    
    # Plot baseline and master model results as horizontal lines
    ax1.axhline(y=avg_baseline_results['full_val_acc'], 
                color='r', linestyle='--', 
                label=f'Baseline (100 epochs)')
    ax1.axhline(y=avg_master_results['full_val_acc'], 
                color='g', linestyle='--', 
                label=f'Master (100 epochs)')
    
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Master Graph Validation Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Neighborhood Graph Validation Accuracy
    for lr in learning_rates:
        accuracies = avg_interactive_results[lr]['neighborhood']
        ax2.plot(epochs, accuracies, marker='o', label=f'Interactive (lr={lr})')
    
    # Plot baseline and master model results as horizontal lines
    ax2.axhline(y=avg_baseline_results['neighborhood_acc'], 
                color='r', linestyle='--', 
                label=f'Baseline (100 epochs)')
    ax2.axhline(y=avg_master_results['neighborhood_acc'], 
                color='g', linestyle='--', 
                label=f'Master (100 epochs)')
    
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Neighborhood Graph Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle('Model Performance Comparison: Master Graph vs Neighborhood Graph (Averaged over {} runs)'.format(num_repeats))
    plt.tight_layout()
    plt.savefig("interactive_training_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 