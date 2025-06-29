import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/Users/huubal/scriptie/data")
sys.path.append("/Users/huubal/scriptie/paperNodes")
from paperNodes_graph import arXivHyperGraph
from model import arXivHGNN  # Assuming arXivHGNN is in your model module


def train(model, data, train_mask, val_mask, epochs, lr, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.embeddings.to(device), data.incidence.to(device))
        loss = criterion(out[train_mask], data.labels[train_mask].to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.embeddings.to(device), data.incidence.to(device))
        pred = out.argmax(dim=1)
        correct = pred[val_mask].eq(data.labels[val_mask].to(device)).sum().item()
        acc = correct / val_mask.sum().item()
    return acc


def get_masks(num_nodes, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    split = int((1 - val_ratio) * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:split]] = True
    val_mask[indices[split:]] = True
    return train_mask, val_mask


def run_outlier_expert_experiment(repeat_id, learning_rates, device='cpu'):
    print(f"\n===== REPEAT {repeat_id + 1} =====")

    # Step 1: Initialize hypergraph and subgraph with dropout and outlier removal
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    subgraph = hypergraph.construct_subgraph(dropout=0.1)
    
    # Store original state before outlier removal
    original_embeddings = subgraph.embeddings.clone()
    original_labels = subgraph.labels.clone()
    original_node_to_authors = copy.deepcopy(subgraph.node_to_authors)
    
    # Remove outliers (z_thresh=3.0 is default)
    print("Removing outliers from subgraph...")
    subgraph.remove_outliers(outlier_fraction=0.1)
    num_nodes_no_outliers = subgraph.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")

    # Step 2: Create masks for training without outliers
    train_mask, val_mask = get_masks(num_nodes_no_outliers)
    original_train_mask = train_mask.clone()
    original_val_mask = val_mask.clone()

    # Step 3: Train baseline model on clean data (no outliers)
    in_dim = subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)
    model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)

    print("Training baseline model on clean data (no outliers)...")
    baseline_acc = train(model, subgraph, train_mask, val_mask, epochs=100, lr=1e-3, device=device)
    print(f"Baseline Accuracy (no outliers): {baseline_acc:.4f}")

    # Save baseline model state and clean subgraph state
    original_model_state = copy.deepcopy(model.state_dict())
    clean_subgraph_state = copy.deepcopy(subgraph)

    # Step 4: Restore outliers for expert training
    print("Restoring outliers for expert training...")
    subgraph.restore_outliers()
    total_nodes_with_outliers = subgraph.embeddings.shape[0]
    num_outliers = total_nodes_with_outliers - num_nodes_no_outliers
    print(f"Total nodes with outliers: {total_nodes_with_outliers}")
    print(f"Number of outliers: {num_outliers}")

    # Step 4.1: Evaluate baseline model performance on outliers before training on them
    print("Evaluating baseline model on outliers before expert training...")
    baseline_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    baseline_model.load_state_dict(copy.deepcopy(original_model_state))
    baseline_model.eval()

    with torch.no_grad():
        out = baseline_model(subgraph.embeddings.to(device), subgraph.incidence.to(device))
        pred = out.argmax(dim=1)

        # Create mask for outliers (last num_outliers nodes)
        outlier_mask = torch.zeros(total_nodes_with_outliers, dtype=torch.bool)
        outlier_mask[-num_outliers:] = True

        correct_outliers = pred[outlier_mask].eq(subgraph.labels[outlier_mask].to(device)).sum().item()
        outlier_acc = correct_outliers / outlier_mask.sum().item()

    print(f"Baseline Accuracy on Outliers (before expert training): {outlier_acc:.4f}")


    # Step 5: Try different learning rates for expert training on outliers
    accuracies_by_lr = {}

    for lr in learning_rates:
        print(f"\n--- Expert Training on Outliers (LR={lr}) ---")
        
        # Reset model to baseline state
        expert_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
        expert_model.load_state_dict(copy.deepcopy(original_model_state))
        
        # Create masks for expert training: only train on outliers
        expert_train_mask = torch.zeros(total_nodes_with_outliers, dtype=torch.bool)
        expert_val_mask = torch.zeros(total_nodes_with_outliers, dtype=torch.bool)
        
        # Set outlier nodes (last num_outliers nodes) as training set
        expert_train_mask[-num_outliers:] = True
        # Validation mask remains empty for expert phase (we'll evaluate on original val set)
        expert_val_mask[-num_outliers:] = True  # Use some outliers for validation during expert training
        
        # Split outliers into train/val for expert phase
        outlier_indices = torch.arange(num_nodes_no_outliers, total_nodes_with_outliers)
        outlier_perm = torch.randperm(len(outlier_indices))
        outlier_split = int(0.8 * len(outlier_indices))
        
        expert_train_mask.fill_(False)
        expert_val_mask.fill_(False)
        expert_train_mask[outlier_indices[outlier_perm[:outlier_split]]] = True
        expert_val_mask[outlier_indices[outlier_perm[outlier_split:]]] = True

        print(f"Expert training on {expert_train_mask.sum().item()} outliers")
        print(f"Expert validation on {expert_val_mask.sum().item()} outliers")

        # Train expert model on outliers
        expert_acc = train(expert_model, subgraph, expert_train_mask, expert_val_mask, 
                          epochs=50, lr=lr, device=device)
        print(f"Expert Training Accuracy (on outliers): {expert_acc:.4f}")

        # Step 6: Evaluate final model on ORIGINAL validation set (without outliers)
        expert_model.eval()
        with torch.no_grad():
            out = expert_model(subgraph.embeddings.to(device), subgraph.incidence.to(device))
            pred = out.argmax(dim=1)
            
            # Create original validation mask for full graph (extend original mask with False for outliers)
            extended_original_val_mask = torch.cat([
                original_val_mask, 
                torch.zeros(num_outliers, dtype=torch.bool)
            ])
            
            correct = pred[extended_original_val_mask].eq(subgraph.labels[extended_original_val_mask].to(device)).sum().item()
            final_acc = correct / extended_original_val_mask.sum().item()
        
        print(f"Final Accuracy on Original Validation Set: {final_acc:.4f}")
        accuracies_by_lr[lr] = {
            'baseline': baseline_acc,
            'baseline_on_outliers': outlier_acc,
            'expert_on_outliers': expert_acc,
            'final_on_original': final_acc
        }

    return accuracies_by_lr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [5e-4, 1e-3, 5e-3]
    num_repeats = 5

    # Initialize results container
    all_runs = {lr: {'baseline': [], 'baseline_on_outliers': [], 'expert_on_outliers': [], 'final_on_original': []} 
            for lr in learning_rates} 

    for repeat in range(num_repeats):
        result = run_outlier_expert_experiment(repeat, learning_rates, device=device)
        for lr in learning_rates:
            all_runs[lr]['baseline'].append(result[lr]['baseline'])
            all_runs[lr]['baseline_on_outliers'].append(result[lr]['baseline_on_outliers'])
            all_runs[lr]['expert_on_outliers'].append(result[lr]['expert_on_outliers'])
            all_runs[lr]['final_on_original'].append(result[lr]['final_on_original']) 

    # Compute averages and standard deviations
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for lr in learning_rates:
        baseline_mean = np.mean(all_runs[lr]['baseline'])
        baseline_std = np.std(all_runs[lr]['baseline'])
        expert_mean = np.mean(all_runs[lr]['expert_on_outliers'])
        expert_std = np.std(all_runs[lr]['expert_on_outliers'])
        final_mean = np.mean(all_runs[lr]['final_on_original'])
        final_std = np.std(all_runs[lr]['final_on_original'])
        outlier_baseline_mean = np.mean(all_runs[lr]['baseline_on_outliers'])
        outlier_baseline_std = np.std(all_runs[lr]['baseline_on_outliers'])

        print(f"\nLearning Rate: {lr}")
        print(f"Baseline (on outliers):         {outlier_baseline_mean:.4f} ± {outlier_baseline_std:.4f}")
        print(f"Baseline (no outliers):      {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"Expert (on outliers):        {expert_mean:.4f} ± {expert_std:.4f}")
        print(f"Final (original val set):    {final_mean:.4f} ± {final_std:.4f}")
        print(f"Improvement over baseline:   {final_mean - baseline_mean:+.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Bar Plot 1: Comparison of all four metrics
    x_pos = np.arange(len(learning_rates))
    width = 0.2

    baseline_means = [np.mean(all_runs[lr]['baseline']) for lr in learning_rates]
    baseline_outlier_means = [np.mean(all_runs[lr]['baseline_on_outliers']) for lr in learning_rates]
    expert_means = [np.mean(all_runs[lr]['expert_on_outliers']) for lr in learning_rates]
    final_means = [np.mean(all_runs[lr]['final_on_original']) for lr in learning_rates]

    baseline_stds = [np.std(all_runs[lr]['baseline']) for lr in learning_rates]
    baseline_outlier_stds = [np.std(all_runs[lr]['baseline_on_outliers']) for lr in learning_rates]
    expert_stds = [np.std(all_runs[lr]['expert_on_outliers']) for lr in learning_rates]
    final_stds = [np.std(all_runs[lr]['final_on_original']) for lr in learning_rates]

    ax1.bar(x_pos - 1.5*width, baseline_means, width, yerr=baseline_stds, 
            label='Baseline (no outliers)', alpha=0.8)
    ax1.bar(x_pos - 0.5*width, baseline_outlier_means, width, yerr=baseline_outlier_stds, 
            label='Baseline (on outliers)', alpha=0.8)
    ax1.bar(x_pos + 0.5*width, expert_means, width, yerr=expert_stds, 
            label='Expert (on outliers)', alpha=0.8)
    ax1.bar(x_pos + 1.5*width, final_means, width, yerr=final_stds, 
            label='Final (original val)', alpha=0.8)

    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Outlier Expert Training Results')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(lr) for lr in learning_rates])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar Plot 2: Improvement over baseline
    improvements = [final_means[i] - baseline_means[i] for i in range(len(learning_rates))]
    improvement_stds = [np.sqrt(baseline_stds[i]**2 + final_stds[i]**2) for i in range(len(learning_rates))]

    bars = ax2.bar(x_pos, improvements, yerr=improvement_stds, alpha=0.8,
                color=['green' if imp > 0 else 'red' for imp in improvements])
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title('Improvement of Final Model over Baseline')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(lr) for lr in learning_rates])
    ax2.grid(True, alpha=0.3)

    # Add value labels to bars
    for i, (bar, imp, std) in enumerate(zip(bars, improvements, improvement_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (std if height > 0 else -std),
                f'{imp:+.4f}', ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    plt.savefig("outlier_expert_training_results.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()