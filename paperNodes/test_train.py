import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/Users/huubal/scriptie/data")
sys.path.append("/Users/huubal/scriptie/paperNodes")
from paperNodes_graph import arXivHyperGraph
from model import arXivHGNN

def train(model, data, epochs, lr, device='cpu'):
    """
    Train the model using the built-in train and validation masks.
    
    Args:
        model: The model to train
        data: The subgraph data containing embeddings, incidence matrix, labels, and masks
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
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

def evaluate_model(model, data, val_mask, device='cpu'):
    """
    Evaluate a model on a specific validation set.
    
    Args:
        model: The model to evaluate
        data: The subgraph data containing embeddings, incidence matrix, and labels
        val_mask: The validation mask to use
        device: Device to evaluate on
    """
    model.eval()
    with torch.no_grad():
        out = model(data.embeddings.to(device), data.incidence.to(device))
        pred = out.argmax(dim=1)
        correct = pred[val_mask].eq(data.labels[val_mask].to(device)).sum().item()
        acc = correct / val_mask.sum().item()
    return acc

def run_experiment(device='cpu'):
    print("\n===== Running Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    original_subgraph = hypergraph.construct_subgraph(dropout=0.1)
    
    # Initialize model parameters
    in_dim = original_subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)

    # Step 2: Train master model on full subgraph
    print("\nTraining master model on full subgraph...")
    master_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    master_acc = train(master_model, original_subgraph, epochs=100, lr=1e-3, device=device)
    print(f"Master model accuracy: {master_acc:.4f}")

    # Step 3: Create a copy of the subgraph, remove outliers and train baseline model
    print("\nRemoving outliers and training baseline model...")
    subgraph_without_outliers = copy.deepcopy(original_subgraph)
    subgraph_without_outliers.remove_outliers(outlier_fraction=0.01)
    num_nodes_no_outliers = subgraph_without_outliers.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")
    
    # Train baseline model using the automatically generated masks
    baseline_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    baseline_acc = train(baseline_model, subgraph_without_outliers, epochs=100, lr=1e-3, device=device)
    print(f"Baseline model accuracy on clean data: {baseline_acc:.4f}")

    # Step 4: Create neighborhood graph for interactive training
    print("\nConstructing neighborhood graph for interactive training...")
    neighborhood_graph = subgraph_without_outliers.construct_outlier_neighbourhood()
    if neighborhood_graph is None:
        print("No outliers found to construct neighborhood graph!")
        return None
    
    num_nodes_neighborhood = neighborhood_graph.embeddings.shape[0]
    print(f"Neighborhood graph size: {num_nodes_neighborhood} nodes")
    
    # Train interactive model using the automatically generated masks
    print("\nTraining interactive model on neighborhood graph...")
    interactive_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    interactive_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
    interactive_acc = train(interactive_model, neighborhood_graph, epochs=15, lr=1e-4, device=device)
    print(f"Interactive model accuracy: {interactive_acc:.4f}")

    # Step 5: Comprehensive evaluation
    print("\n===== Model Evaluation =====")
    
    # Evaluate on master model's validation set
    print("\nEvaluating on master model's validation set:")
    master_val_acc = evaluate_model(master_model, original_subgraph, original_subgraph.val_mask, device)
    baseline_val_acc = evaluate_model(baseline_model, original_subgraph, original_subgraph.val_mask, device)
    interactive_val_acc = evaluate_model(interactive_model, original_subgraph, original_subgraph.val_mask, device)
    
    print(f"Master model accuracy: {master_val_acc:.4f}")
    print(f"Baseline model accuracy: {baseline_val_acc:.4f}")
    print(f"Interactive model accuracy: {interactive_val_acc:.4f}")
    
    # Evaluate on interactive model's validation set
    print("\nEvaluating on interactive model's validation set:")
    master_neighborhood_acc = evaluate_model(master_model, neighborhood_graph, neighborhood_graph.val_mask, device)
    baseline_neighborhood_acc = evaluate_model(baseline_model, neighborhood_graph, neighborhood_graph.val_mask, device)
    interactive_neighborhood_acc = evaluate_model(interactive_model, neighborhood_graph, neighborhood_graph.val_mask, device)
    
    print(f"Master model accuracy: {master_neighborhood_acc:.4f}")
    print(f"Baseline model accuracy: {baseline_neighborhood_acc:.4f}")
    print(f"Interactive model accuracy: {interactive_neighborhood_acc:.4f}")
    
    # Print summary of improvements
    print("\n===== Performance Summary =====")
    print("On master validation set:")
    print(f"Baseline vs Master: {baseline_val_acc - master_val_acc:+.4f}")
    print(f"Interactive vs Master: {interactive_val_acc - master_val_acc:+.4f}")
    print(f"Interactive vs Baseline: {interactive_val_acc - baseline_val_acc:+.4f}")
    
    print("\nOn neighborhood validation set:")
    print(f"Baseline vs Master: {baseline_neighborhood_acc - master_neighborhood_acc:+.4f}")
    print(f"Interactive vs Master: {interactive_neighborhood_acc - master_neighborhood_acc:+.4f}")
    print(f"Interactive vs Baseline: {interactive_neighborhood_acc - baseline_neighborhood_acc:+.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiment(device=device)

if __name__ == "__main__":
    main() 