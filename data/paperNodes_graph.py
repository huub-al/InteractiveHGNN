"""
ArXiv Hypergraph Data Classes

This module provides classes for constructing and manipulating hypergraphs
from arXiv paper data. It includes functionality for embedding papers,
tracking author relationships, and dynamically updating graph structure
with real, plausible, and fake papers.
"""

import os
import re
import gzip
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel


class arXivHyperGraph:
    """
    A hypergraph representation of arXiv papers and their author relationships.
    
    This class loads arXiv paper data, computes embeddings for paper abstracts,
    and maintains the relationships between papers and authors. It serves as the
    base graph from which subgraphs can be constructed.
    
    Attributes:
        data_path (str): Path to the arXiv data file (gzipped JSON)
        model_name (str): Name of the pretrained model for embeddings
        cache_path (str): Path to save/load cached graph data
        device (torch.device): Device for computation (CPU or CUDA)
        x (torch.Tensor): Paper embeddings matrix
        y (torch.Tensor): Paper labels tensor
        paper_ids (list): List of arXiv paper IDs
        node_to_authors (dict): Mapping from node indices to author lists
        author_pool (list): List of all unique authors
        author_mean (float): Average number of authors per paper
        paper_id_to_idx (dict): Mapping from paper IDs to node indices
        label_map (dict): Mapping from category names to label indices
        synthetic_labels (dict): Mapping from synthetic label names to indices
        full_label_map (dict): Combined mapping of real and synthetic labels
        train_mask (torch.Tensor): Boolean mask for training nodes
        val_mask (torch.Tensor): Boolean mask for validation nodes
    """
    
    def __init__(self, data_path="arxiv-data/subset_cs_2000.json.gz",
                 model_name="allenai/scibert_scivocab_uncased",
                 cache_path="arxiv_hypergraph.pt"):
        """
        Initialize the arXiv hypergraph.
        
        Args:
            data_path (str): Path to the arXiv data file (gzipped JSON)
            model_name (str): Name of the pretrained model for embeddings
            cache_path (str): Path to save/load cached graph data
        """
        self.data_path = data_path
        self.model_name = model_name
        self.cache_path = cache_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(self.cache_path):
            print(f"Loading cached graph from {self.cache_path}")
            self._load_cache()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self._build()
            self._save_cache()
            
        # Generate train/val masks with 80/20 split
        self._generate_masks()

    def _generate_masks(self):
        """
        Generate training and validation masks with an 80/20 split.
        """
        num_nodes = len(self.paper_ids)
        indices = torch.randperm(num_nodes)
        train_size = int(0.8 * num_nodes)
        
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[indices[:train_size]] = True
        self.val_mask[indices[train_size:]] = True

    def _build(self):
        """
        Build the hypergraph from arXiv data.
        
        This method loads papers, extracts abstracts and authors, computes
        embeddings, and constructs the necessary data structures for the graph.
        """
        papers = self._load_papers()
        abstracts, paper_ids, labels, paper_to_authors = [], [], [], {}

        # Create base label mapping from observed category labels
        all_categories = sorted({p["categories"][0] for p in papers if p.get("abstract")})
        self.label_map = {cat: i for i, cat in enumerate(all_categories)}
        
        # Define synthetic labels with fixed indices to avoid confusion
        self.synthetic_labels = {
            "not a likely collaboration": len(self.label_map),
            "not scientific work": len(self.label_map) + 1
        }
        self.full_label_map = {**self.label_map, **self.synthetic_labels}
        
        # Create reverse mapping for debugging and validation
        self.idx_to_label = {idx: label for label, idx in self.full_label_map.items()}

        for paper in papers:
            if not paper.get("abstract"):
                continue
            authors = self._filter_authors(paper["authors"])
            if not authors:
                continue
            abstracts.append(paper["abstract"])
            paper_ids.append(paper["id"])
            labels.append(self.label_map[paper["categories"][0]])
            paper_to_authors[paper["id"]] = authors

        self.x = self._get_embeddings(abstracts)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.paper_ids = paper_ids
        self.node_to_authors = {i: paper_to_authors[pid] for i, pid in enumerate(paper_ids)}
        self.author_pool = list({a for authors in paper_to_authors.values() for a in authors})
        self.author_mean = np.mean([len(authors) for authors in paper_to_authors.values()])
        self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}

    def _print_summary(self):
        """
        Print a summary of the hypergraph statistics.
        """
        print("\n--- Hypergraph Summary ---")
        print(f"Total papers: {len(self.paper_ids)}")
        print(f"Embedding shape: {self.x.shape}")
        print(f"Number of classes: {len(self.label_map)}")
        print(f"Average authors per paper: {self.author_mean:.2f}")

    def _save_cache(self):
        """
        Save the hypergraph to a cache file for faster loading.
        """
        torch.save({
            "x": self.x,
            "y": self.y,
            "paper_ids": self.paper_ids,
            "node_to_authors": self.node_to_authors,
            "author_mean": self.author_mean,
            "author_pool": self.author_pool,
            "paper_id_to_idx": self.paper_id_to_idx,
            "label_map": self.label_map,
            "synthetic_labels": self.synthetic_labels
        }, self.cache_path)

    def _load_cache(self):
        """
        Load the hypergraph from a cached file.
        """
        state = torch.load(self.cache_path, weights_only=False)
        self.x = state["x"]
        self.y = state["y"]
        self.paper_ids = state["paper_ids"]
        self.node_to_authors = state["node_to_authors"]
        self.author_mean = state["author_mean"]
        self.author_pool = state["author_pool"]
        self.paper_id_to_idx = state["paper_id_to_idx"]
        self.label_map = state["label_map"]
        self.synthetic_labels = state["synthetic_labels"]
        self.full_label_map = {**self.label_map, **self.synthetic_labels}
        
        # Create reverse mapping for debugging and validation
        self.idx_to_label = {idx: label for label, idx in self.full_label_map.items()}

    def _save_cache(self):
        """
        Save the hypergraph to a cache file for faster loading.
        """
        torch.save({
            "x": self.x,
            "y": self.y,
            "paper_ids": self.paper_ids,
            "node_to_authors": self.node_to_authors,
            "author_mean": self.author_mean,
            "author_pool": self.author_pool,
            "paper_id_to_idx": self.paper_id_to_idx,
            "label_map": self.label_map,
            "synthetic_labels": self.synthetic_labels
        }, self.cache_path)

    def _load_papers(self):
        """
        Load papers from the gzipped JSON file.
        
        Returns:
            list: List of paper dictionaries
        """
        with gzip.open(self.data_path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _filter_authors(self, author_string):
        """
        Filter and clean author names from the author string.
        
        Args:
            author_string (str): Raw author string from the paper data
            
        Returns:
            list: List of cleaned author names
        """
        authors = re.split(r',\s*|\s+and\s+', author_string)
        filtered = []
        for author in authors:
            a = author.strip().lower()
            if any(word in a for word in ['university', 'institute', 'lab', 'center', 'department']):
                continue
            if a in ['usa', 'uk', 'china', 'japan', 'germany', 'et al']:
                continue
            if re.match(r'^[a-zA-Z.\- ]+$', author) and len(author.split()) <= 3:
                filtered.append(author.strip())
        return filtered

    def _get_embeddings(self, texts):
        """
        Compute embeddings for paper abstracts using the pretrained model.
        
        Args:
            texts (list): List of paper abstract texts
            
        Returns:
            torch.Tensor: Tensor of paper embeddings
        """
        embeddings = []
        for text in tqdm(texts, desc="Embedding abstracts"):
            inputs = self.tokenizer(text, truncation=True, padding=True,
                                    max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            embeddings.append(cls)
        return torch.stack(embeddings)

    def construct_subgraph(self, dropout=0.0):
        """
        Construct a subgraph from the full hypergraph.
        
        Args:
            dropout (float): Fraction of papers to exclude from the subgraph
            
        Returns:
            arXivSubGraph: A subgraph object containing the selected papers
        """
        mask = np.random.rand(len(self.paper_ids)) > dropout
        indices = np.where(mask)[0]
        return arXivSubGraph(self, indices)


class arXivSubGraph:
    """
    A subgraph of the arXiv hypergraph that can be dynamically modified.
    
    This class maintains a subset of papers from the full hypergraph and
    provides methods to add real, plausible, and fake papers, as well as
    to remove fake papers. It maintains the incidence matrix representing
    the author-paper relationships.
    
    Attributes:
        hypergraph (arXivHyperGraph): Reference to the parent hypergraph
        indices (list): List of indices in the parent graph
        embeddings (torch.Tensor): Paper embeddings matrix for the subgraph
        labels (torch.Tensor): Paper labels tensor for the subgraph
        node_to_authors (dict): Mapping from node indices to author lists
        incidence (torch.Tensor): Sparse incidence matrix of author-paper relationships
        train_mask (torch.Tensor): Boolean mask for training nodes in subgraph
        val_mask (torch.Tensor): Boolean mask for validation nodes in subgraph
    """
    
    def __init__(self, hypergraph: arXivHyperGraph, indices, outlier_only_masks=False, outlier_indices=None):
        """
        Initialize the subgraph from a subset of the full hypergraph.
        
        Args:
            hypergraph (arXivHyperGraph): The parent hypergraph
            indices (list or array): Indices of papers to include in the subgraph
            outlier_only_masks (bool): If True, only outliers will be included in train/val masks
            outlier_indices (list): List of main graph indices that are outliers (for neighborhoods)
        """
        self.hypergraph = hypergraph
        self.indices = list(indices)
        self.sub_idx_to_main = {i: idx for i, idx in enumerate(indices)}
        self.outlier_only_masks = outlier_only_masks

        # Extract relevant data for the subgraph
        self.embeddings = hypergraph.x[indices]
        self.labels = hypergraph.y[indices].clone()
        self.node_to_authors = {i: hypergraph.node_to_authors[idx] for i, idx in enumerate(indices)}

        # Build the initial incidence matrix
        self._rebuild_incidence()

        self.outliers = []  # Track removed outlier indices (relative to subgraph)
        self.main_outliers_idx = outlier_indices if outlier_indices is not None else []
        self.removed_outliers_data = {}  # Maps index to (embedding, label, authors)

        self.generate_masks()

    def _rebuild_incidence(self):
        """
        Rebuild the incidence matrix for the current state of the subgraph.
        
        This creates a sparse matrix where rows represent papers and columns
        represent authors. A value of 1 indicates that an author is associated
        with a paper.
        """
        # Create mapping from authors to papers
        author_to_papers = defaultdict(list)
        for idx, authors in self.node_to_authors.items():
            for author in authors:
                author_to_papers[author].append(idx)

        # Create the incidence matrix
        num_nodes = len(self.node_to_authors)
        num_edges = len(author_to_papers)
        incidence = torch.zeros((num_nodes, num_edges), dtype=torch.float32)

        # Fill the incidence matrix
        for j, (author, papers) in enumerate(author_to_papers.items()):
            for i in papers:
                incidence[i, j] = 1
                
        # Convert to sparse format for efficiency
        self.incidence = incidence.to_sparse_coo()

    def remove_outliers(self, outlier_fraction=0.01):
        """
        Remove papers that are statistical outliers based on embedding distance from the mean.
        
        Args:
            outlier_fraction (float): Fraction of papers to remove as outliers (default: 0.01 for 1%).
        """
        if len(self.embeddings) == 0:
            print("Warning: No embeddings to process for outlier removal.")
            return
        
        # Calculate distances from the mean embedding
        mean = self.embeddings.mean(dim=0)
        distances = torch.norm(self.embeddings - mean, dim=1)
        
        # Determine the number of outliers to remove
        num_outliers = max(1, int(len(self.embeddings) * outlier_fraction))
        
        # Get indices of the papers with largest distances (outliers)
        _, outlier_indices = torch.topk(distances, num_outliers, largest=True)
        outlier_indices = outlier_indices.tolist()
        
        # Store main graph indices of outliers before updating sub_idx_to_main
        main_outlier_indices = [self.sub_idx_to_main[i] for i in outlier_indices]
        self.main_outliers_idx.extend(main_outlier_indices)
        self.outliers.extend(outlier_indices)
        
        # Create list of indices to keep
        keep_indices = [i for i in range(len(self.embeddings)) if i not in outlier_indices]
        
        # Update sub_idx_to_main mapping
        new_sub_idx_to_main = {}
        for new_i, old_i in enumerate(keep_indices):
            new_sub_idx_to_main[new_i] = self.sub_idx_to_main[old_i]
        self.sub_idx_to_main = new_sub_idx_to_main
        
        print(f"Removing {len(outlier_indices)} outliers out of {len(self.embeddings)} total papers ({len(outlier_indices)/len(self.embeddings)*100:.1f}%)")
        
        # Save outlier data for potential restoration
        for i in outlier_indices:
            self.removed_outliers_data[i] = (
                self.embeddings[i].clone(),  # Clone to avoid reference issues
                self.labels[i].clone(),
                self.node_to_authors[i].copy(),  # Copy the author list
            )
        
        # Keep only non-outlier data
        self.embeddings = self.embeddings[keep_indices]
        self.labels = self.labels[keep_indices]
        
        # Rebuild node_to_authors mapping with new consecutive indices
        new_node_to_authors = {}
        for new_i, old_i in enumerate(keep_indices):
            new_node_to_authors[new_i] = self.node_to_authors[old_i]  
        self.node_to_authors = new_node_to_authors
        
        # Rebuild the incidence matrix
        self._rebuild_incidence()
        
        # Regenerate masks after removing outliers
        self.generate_masks()

    def construct_outlier_neighbourhood(self, outlier_masks):
        """
        Construct a subgraph containing only the outliers and their one-hop neighbors.
        
        Returns:
            arXivSubGraph: A new subgraph containing only the outliers and their neighbors,
                        or None if no outliers exist.
        """
        if not self.outliers:
            print("No outliers to construct neighborhood from.")
            return None
            
        # Get all authors from outliers (from saved data)
        outlier_authors = set()
        for old_idx in self.outliers:
            _, _, authors = self.removed_outliers_data[old_idx]
            outlier_authors.update(authors)
            
        # Find neighbors in the CURRENT subgraph that share authors with outliers
        neighbor_indices = set()
        for idx, authors in self.node_to_authors.items():
            if any(author in outlier_authors for author in authors):
                neighbor_indices.add(self.sub_idx_to_main[idx])  # Convert to main graph index
                
        # Combine outliers and neighbors
        all_indices = set()
        
        # Add outlier indices (from main graph)
        all_indices.update(self.main_outliers_idx)
        
        # Add neighbor indices  
        all_indices.update(neighbor_indices)
            
        # Convert to sorted list for consistency
        all_indices = sorted(list(all_indices))
        
        print(f"Constructing outlier neighborhood with {len(self.main_outliers_idx)} outliers and {len(neighbor_indices)} neighbors (total: {len(all_indices)})")
        print(f"Outlier percentage: {len(self.main_outliers_idx)/len(all_indices)*100:.1f}%")
        
        # Create a new subgraph with these indices and outlier-only masks
        return arXivSubGraph(self.hypergraph, all_indices, outlier_only_masks=outlier_masks, outlier_indices=self.main_outliers_idx)

    def generate_masks(self):
        """
        Generate training and validation masks for the subgraph based on the parent graph's masks.
        
        If outlier_only_masks is True, only outliers will be included in train/val masks.
        Otherwise, all nodes will be included based on their parent graph assignments.
        """
        num_nodes = len(self.embeddings)  # Use embeddings length to get current number of nodes
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        if self.outlier_only_masks:
            # Only include outliers in the masks
            for i, parent_idx in self.sub_idx_to_main.items():
                # Check if this node is an outlier (in main_outliers_idx)
                if parent_idx in self.main_outliers_idx:
                    if self.hypergraph.train_mask[parent_idx]:
                        self.train_mask[i] = True
                    elif self.hypergraph.val_mask[parent_idx]:
                        self.val_mask[i] = True
        else:
            # Include all nodes based on their parent graph assignments
            for i, parent_idx in self.sub_idx_to_main.items():
                if self.hypergraph.train_mask[parent_idx]:
                    self.train_mask[i] = True
                elif self.hypergraph.val_mask[parent_idx]:
                    self.val_mask[i] = True

    def generate_neighbourhoods(self, outliers_per_subgraph=25):
        """
        Generate a list of subgraphs containing outliers and their one-hop neighbors.
        Outliers are evenly distributed across neighborhoods, with validation outliers
        distributed evenly across all neighborhoods.
        
        Args:
            outliers_per_subgraph (int): Number of outliers to include in each subgraph (default: 25)
            
        Returns:
            list: List of arXivSubGraph objects, each containing outliers and their neighbors
        """
        if not self.outliers:
            print("No outliers to generate neighborhoods from.")
            return []
        
        # Get all outlier authors from the removed outliers data
        outlier_authors = set()
        for old_idx in self.outliers:
            _, _, authors = self.removed_outliers_data[old_idx]
            outlier_authors.update(authors)
        
        # Find all neighbors in the CURRENT subgraph that share authors with outliers
        neighbor_indices = set()
        for idx, authors in self.node_to_authors.items():
            if any(author in outlier_authors for author in authors):
                neighbor_indices.add(self.sub_idx_to_main[idx])  # Convert to main graph index
        
        # Group outliers by their train/val status in the parent graph
        train_outliers = []
        val_outliers = []
        
        for old_idx in self.outliers:
            # Get the main graph index for this outlier
            main_idx = self.main_outliers_idx[self.outliers.index(old_idx)]
            
            if self.hypergraph.train_mask[main_idx]:
                train_outliers.append(old_idx)
            elif self.hypergraph.val_mask[main_idx]:
                val_outliers.append(old_idx)
        
        print(f"Found {len(train_outliers)} training outliers and {len(val_outliers)} validation outliers")
        
        # Calculate number of neighborhoods needed
        total_outliers = len(train_outliers) + len(val_outliers)
        num_neighborhoods = max(1, total_outliers // outliers_per_subgraph)
        print(f"Creating {num_neighborhoods} neighborhoods with approximately {outliers_per_subgraph} outliers each")
        
        # Evenly divide training outliers
        train_per_neigh = len(train_outliers) // num_neighborhoods
        train_remainder = len(train_outliers) % num_neighborhoods
        # Evenly divide validation outliers
        val_per_neigh = len(val_outliers) // num_neighborhoods
        val_remainder = len(val_outliers) % num_neighborhoods
        
        subgraphs = []
        train_used = 0
        val_used = 0
        for i in range(num_neighborhoods):
            this_train = train_per_neigh + (1 if i < train_remainder else 0)
            this_val = val_per_neigh + (1 if i < val_remainder else 0)
            selected_train = train_outliers[train_used:train_used+this_train]
            selected_val = val_outliers[val_used:val_used+this_val]
            selected_outliers = selected_train + selected_val
            if not selected_outliers:
                continue
            train_used += this_train
            val_used += this_val
            # Get authors from selected outliers
            selected_authors = set()
            for old_idx in selected_outliers:
                _, _, authors = self.removed_outliers_data[old_idx]
                selected_authors.update(authors)
            # Find neighbors that share authors with selected outliers
            subgraph_neighbors = set()
            for idx, authors in self.node_to_authors.items():
                if any(author in selected_authors for author in authors):
                    subgraph_neighbors.add(self.sub_idx_to_main[idx])
            # Combine selected outliers and their neighbors
            all_indices = set()
            selected_outlier_main_indices = [self.main_outliers_idx[self.outliers.index(old_idx)] for old_idx in selected_outliers]
            all_indices.update(selected_outlier_main_indices)
            all_indices.update(subgraph_neighbors)
            all_indices = sorted(list(all_indices))
            print(f"Creating subgraph {i+1}/{num_neighborhoods} with {len(selected_outliers)} outliers ({len(selected_train)} train, {len(selected_val)} val) and {len(subgraph_neighbors)} neighbors (total: {len(all_indices)})")
            subgraph = arXivSubGraph(self.hypergraph, all_indices, outlier_only_masks=True, outlier_indices=selected_outlier_main_indices)
            subgraphs.append(subgraph)
        print(f"Generated {len(subgraphs)} outlier neighborhood subgraphs")
        return subgraphs