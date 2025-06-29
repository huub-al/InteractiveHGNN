import torch
import unittest
import sys
sys.path.append("/Users/huubal/scriptie/data")
sys.path.append("/Users/huubal/scriptie/paperNodes")
from paperNodes_graph import arXivHyperGraph, arXivSubGraph

class TestArXivHyperGraph(unittest.TestCase):
    """Focused test suite for arXiv hypergraph functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize the main hypergraph with a smaller dataset for faster testing
        self.hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_2000.json.gz")
    
    def test_embeddings_correspondence(self):
        """Test that embeddings in subgraphs correspond correctly to main graph."""
        # Create a subgraph with 50% of the nodes
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        
        # Check that embeddings match exactly between subgraph and main graph
        for sub_idx, main_idx in subgraph.sub_idx_to_main.items():
            self.assertTrue(
                torch.allclose(
                    subgraph.embeddings[sub_idx],
                    self.hypergraph.x[main_idx],
                    rtol=1e-5,
                    atol=1e-5
                ),
                f"Embedding mismatch: Node {main_idx} has different embeddings in subgraph vs main graph"
            )
    
    def test_validation_nodes_preserved(self):
        """Test that validation nodes remain validation nodes in subgraphs."""
        # Create a subgraph
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        
        # Check that validation nodes in subgraph are also validation nodes in main graph
        subgraph_val_indices = torch.where(subgraph.val_mask)[0]
        for sub_idx in subgraph_val_indices:
            main_idx = subgraph.sub_idx_to_main[sub_idx.item()]
            self.assertTrue(
                self.hypergraph.val_mask[main_idx],
                f"Data leakage: Node {main_idx} is validation in subgraph but not in main graph"
            )
    
    def test_train_nodes_preserved(self):
        """Test that training nodes remain training nodes in subgraphs."""
        # Create a subgraph
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        
        # Check that training nodes in subgraph are also training nodes in main graph
        subgraph_train_indices = torch.where(subgraph.train_mask)[0]
        for sub_idx in subgraph_train_indices:
            main_idx = subgraph.sub_idx_to_main[sub_idx.item()]
            self.assertTrue(
                self.hypergraph.train_mask[main_idx],
                f"Data leakage: Node {main_idx} is training in subgraph but not in main graph"
            )
    
    def test_outlier_removal_embeddings_consistency(self):
        """Test that outlier removal maintains embedding consistency."""
        # Create initial subgraph
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        
        # Remove outliers
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Check that remaining embeddings still correspond to main graph
        for sub_idx, main_idx in subgraph.sub_idx_to_main.items():
            self.assertTrue(
                torch.allclose(
                    subgraph.embeddings[sub_idx],
                    self.hypergraph.x[main_idx],
                    rtol=1e-5,
                    atol=1e-5
                ),
                f"Embedding mismatch after outlier removal: Node {main_idx}"
            )
    
    def test_outlier_removal_validation_preservation(self):
        """Test that outlier removal preserves validation node status."""
        # Create initial subgraph
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        
        # Remove outliers
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Check that validation nodes are still validation nodes
        subgraph_val_indices = torch.where(subgraph.val_mask)[0]
        for sub_idx in subgraph_val_indices:
            main_idx = subgraph.sub_idx_to_main[sub_idx.item()]
            self.assertTrue(
                self.hypergraph.val_mask[main_idx],
                f"Validation node {main_idx} lost validation status after outlier removal"
            )
    
    def test_neighborhood_embeddings_consistency(self):
        """Test that generated neighborhoods maintain embedding consistency."""
        # Create subgraph and remove outliers
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Generate neighborhoods
        neighborhoods = subgraph.generate_neighbourhoods()
        
        if len(neighborhoods) > 0:
            for i, neighborhood in enumerate(neighborhoods):
                # Check embeddings in this neighborhood
                for sub_idx, main_idx in neighborhood.sub_idx_to_main.items():
                    self.assertTrue(
                        torch.allclose(
                            neighborhood.embeddings[sub_idx],
                            self.hypergraph.x[main_idx],
                            rtol=1e-5,
                            atol=1e-5
                        ),
                        f"Embedding mismatch in neighborhood {i}: Node {main_idx}"
                    )
        else:
            self.skipTest("No neighborhoods were generated")
    
    def test_neighborhood_validation_preservation(self):
        """Test that generated neighborhoods preserve validation node status."""
        # Create subgraph and remove outliers
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Generate neighborhoods
        neighborhoods = subgraph.generate_neighbourhoods()
        
        if len(neighborhoods) > 0:
            for i, neighborhood in enumerate(neighborhoods):
                # Check validation nodes in this neighborhood
                subgraph_val_indices = torch.where(neighborhood.val_mask)[0]
                for sub_idx in subgraph_val_indices:
                    main_idx = neighborhood.sub_idx_to_main[sub_idx.item()]
                    self.assertTrue(
                        self.hypergraph.val_mask[main_idx],
                        f"Validation node {main_idx} lost validation status in neighborhood {i}"
                    )
        else:
            self.skipTest("No neighborhoods were generated")
    
    def test_neighborhood_outlier_only_masks(self):
        """Test that generated neighborhoods only include outliers in train/val masks."""
        # Create subgraph and remove outliers
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Generate neighborhoods
        neighborhoods = subgraph.generate_neighbourhoods()
        
        if len(neighborhoods) > 0:
            for i, neighborhood in enumerate(neighborhoods):
                # Check that the neighborhood has outlier_only_masks enabled
                self.assertTrue(
                    neighborhood.outlier_only_masks,
                    f"Neighborhood {i} should have outlier_only_masks=True"
                )
                
                # Get all nodes in train and val masks
                train_indices = torch.where(neighborhood.train_mask)[0]
                val_indices = torch.where(neighborhood.val_mask)[0]
                all_masked_indices = torch.cat([train_indices, val_indices])
                
                # Check that all masked nodes are outliers
                for sub_idx in all_masked_indices:
                    main_idx = neighborhood.sub_idx_to_main[sub_idx.item()]
                    self.assertIn(
                        main_idx,
                        subgraph.main_outliers_idx,
                        f"Node {main_idx} in neighborhood {i} is in train/val mask but is not an outlier"
                    )
                
                # Check that all outliers in this neighborhood are in masks
                neighborhood_outlier_indices = []
                for sub_idx, main_idx in neighborhood.sub_idx_to_main.items():
                    if main_idx in subgraph.main_outliers_idx:
                        neighborhood_outlier_indices.append(sub_idx)
                
                for sub_idx in neighborhood_outlier_indices:
                    self.assertTrue(
                        neighborhood.train_mask[sub_idx] or neighborhood.val_mask[sub_idx],
                        f"Outlier node {neighborhood.sub_idx_to_main[sub_idx]} in neighborhood {i} is not in any mask"
                    )
                
                print(f"Neighborhood {i}: {len(train_indices)} train outliers, {len(val_indices)} val outliers, "
                      f"{len(neighborhood.sub_idx_to_main) - len(all_masked_indices)} non-outlier nodes (unmasked)")
        else:
            self.skipTest("No neighborhoods were generated")
    
    def test_regular_subgraph_masks(self):
        """Test that regular subgraphs include all nodes in masks (not outlier-only)."""
        # Create a regular subgraph
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        
        # Check that it doesn't have outlier_only_masks
        self.assertFalse(subgraph.outlier_only_masks)
        
        # Check that all nodes are in either train or val mask
        all_masked = torch.logical_or(subgraph.train_mask, subgraph.val_mask)
        self.assertEqual(
            torch.sum(all_masked).item(),
            len(subgraph.embeddings),
            "Regular subgraph should include all nodes in train/val masks"
        )
    
    def test_outlier_only_subgraph_masks(self):
        """Test that subgraphs with outlier_only_masks=True only include outliers in masks."""
        # Create a subgraph and remove outliers
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Create a new subgraph with outlier_only_masks=True
        outlier_subgraph = arXivSubGraph(
            self.hypergraph, 
            list(subgraph.sub_idx_to_main.values()), 
            outlier_only_masks=True
        )
        
        # Check that it has outlier_only_masks enabled
        self.assertTrue(outlier_subgraph.outlier_only_masks)
        
        # Check that only outliers are in masks
        train_indices = torch.where(outlier_subgraph.train_mask)[0]
        val_indices = torch.where(outlier_subgraph.val_mask)[0]
        all_masked_indices = torch.cat([train_indices, val_indices])
        
        for sub_idx in all_masked_indices:
            main_idx = outlier_subgraph.sub_idx_to_main[sub_idx.item()]
            self.assertIn(
                main_idx,
                subgraph.main_outliers_idx,
                f"Node {main_idx} is in mask but is not an outlier"
            )
        
        # Check that all outliers are in masks
        for sub_idx, main_idx in outlier_subgraph.sub_idx_to_main.items():
            if main_idx in subgraph.main_outliers_idx:
                self.assertTrue(
                    outlier_subgraph.train_mask[sub_idx] or outlier_subgraph.val_mask[sub_idx],
                    f"Outlier node {main_idx} is not in any mask"
                )
    
    def test_construct_outlier_neighbourhood_masks(self):
        """Test that construct_outlier_neighbourhood creates subgraphs with outlier-only masks."""
        # Create a subgraph and remove outliers
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Construct outlier neighborhood with outlier-only masks
        neighborhood = subgraph.construct_outlier_neighbourhood(True)  # True for outlier-only masks
        
        if neighborhood is not None:
            # Check that it has outlier_only_masks enabled
            self.assertTrue(
                neighborhood.outlier_only_masks,
                "construct_outlier_neighbourhood should create subgraphs with outlier_only_masks=True"
            )
            
            # Get all nodes in train and val masks
            train_indices = torch.where(neighborhood.train_mask)[0]
            val_indices = torch.where(neighborhood.val_mask)[0]
            all_masked_indices = torch.cat([train_indices, val_indices])
            
            # Check that all masked nodes are outliers
            for sub_idx in all_masked_indices:
                main_idx = neighborhood.sub_idx_to_main[sub_idx.item()]
                self.assertIn(
                    main_idx,
                    subgraph.main_outliers_idx,
                    f"Node {main_idx} in outlier neighborhood is in train/val mask but is not an outlier"
                )
            
            # Check that all outliers in this neighborhood are in masks
            neighborhood_outlier_indices = []
            for sub_idx, main_idx in neighborhood.sub_idx_to_main.items():
                if main_idx in subgraph.main_outliers_idx:
                    neighborhood_outlier_indices.append(sub_idx)
            
            for sub_idx in neighborhood_outlier_indices:
                self.assertTrue(
                    neighborhood.train_mask[sub_idx] or neighborhood.val_mask[sub_idx],
                    f"Outlier node {neighborhood.sub_idx_to_main[sub_idx]} in outlier neighborhood is not in any mask"
                )
            
            print(f"Outlier neighborhood: {len(train_indices)} train outliers, {len(val_indices)} val outliers, "
                  f"{len(neighborhood.sub_idx_to_main) - len(all_masked_indices)} non-outlier nodes (unmasked)")
        else:
            self.skipTest("No outliers available to construct neighborhood")
    
    def test_neighborhood_completeness(self):
        """Test that all neighbors of outliers are included in the generated neighborhoods."""
        # Create subgraph and remove outliers
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Generate neighborhoods
        neighborhoods = subgraph.generate_neighbourhoods()
        
        if len(neighborhoods) > 0:
            # Get all outlier authors from the removed outliers data
            outlier_authors = set()
            for old_idx in subgraph.outliers:
                _, _, authors = subgraph.removed_outliers_data[old_idx]
                outlier_authors.update(authors)
            
            # Find all neighbors in the original subgraph that share authors with outliers
            expected_neighbor_indices = set()
            for idx, authors in subgraph.node_to_authors.items():
                if any(author in outlier_authors for author in authors):
                    expected_neighbor_indices.add(subgraph.sub_idx_to_main[idx])
            
            # Collect all nodes from all neighborhoods
            all_neighborhood_nodes = set()
            for neighborhood in neighborhoods:
                all_neighborhood_nodes.update(neighborhood.sub_idx_to_main.values())
            
            # Check that all expected neighbors are included in at least one neighborhood
            missing_neighbors = expected_neighbor_indices - all_neighborhood_nodes
            self.assertEqual(
                len(missing_neighbors), 
                0, 
                f"Missing {len(missing_neighbors)} neighbors of outliers in generated neighborhoods: {missing_neighbors}"
            )
            
            print(f"All {len(expected_neighbor_indices)} neighbors of outliers are included in the {len(neighborhoods)} neighborhoods")
        else:
            self.skipTest("No neighborhoods were generated")
    
    def test_large_vs_small_neighborhoods_completeness(self):
        """Test that every node in the large outlier neighborhood occurs in at least one small neighborhood."""
        # Create subgraph and remove outliers
        subgraph = self.hypergraph.construct_subgraph(dropout=0.5)
        subgraph.remove_outliers(outlier_fraction=0.1)
        
        # Construct large neighborhood graph
        large_neighborhood = subgraph.construct_outlier_neighbourhood(False)  # Use False to include all nodes, not just outliers
        
        if large_neighborhood is None:
            self.skipTest("No outliers available to construct large neighborhood")
        
        # Generate small neighborhoods
        small_neighborhoods = subgraph.generate_neighbourhoods()
        
        if len(small_neighborhoods) == 0:
            self.skipTest("No small neighborhoods were generated")
        
        # Get all nodes from the large neighborhood
        large_neighborhood_nodes = set(large_neighborhood.sub_idx_to_main.values())
        
        # Collect all nodes from all small neighborhoods
        all_small_neighborhood_nodes = set()
        for neighborhood in small_neighborhoods:
            all_small_neighborhood_nodes.update(neighborhood.sub_idx_to_main.values())
        
        # Check that every node in the large neighborhood is also in at least one small neighborhood
        missing_nodes = large_neighborhood_nodes - all_small_neighborhood_nodes
        self.assertEqual(
            len(missing_nodes), 
            0, 
            f"Missing {len(missing_nodes)} nodes from large neighborhood in small neighborhoods: {missing_nodes}"
        )
        
        # Check that every node in small neighborhoods is also in the large neighborhood
        extra_nodes = all_small_neighborhood_nodes - large_neighborhood_nodes
        self.assertEqual(
            len(extra_nodes), 
            0, 
            f"Found {len(extra_nodes)} extra nodes in small neighborhoods not present in large neighborhood: {extra_nodes}"
        )
        
        # Verify exact equality
        self.assertEqual(
            large_neighborhood_nodes, 
            all_small_neighborhood_nodes,
            "Large neighborhood and union of small neighborhoods should contain exactly the same nodes"
        )
        
        print(f"Large neighborhood: {len(large_neighborhood_nodes)} nodes")
        print(f"Union of small neighborhoods: {len(all_small_neighborhood_nodes)} nodes")
        print(f"Number of small neighborhoods: {len(small_neighborhoods)}")
        print("✓ Large neighborhood and union of small neighborhoods contain exactly the same nodes")
        
        # Additional check: verify that the total number of nodes across all small neighborhoods
        # equals the number of nodes in the large neighborhood (accounting for overlaps)
        total_nodes_in_small_neighborhoods = sum(len(neighborhood.sub_idx_to_main) for neighborhood in small_neighborhoods)
        print(f"Total nodes across all small neighborhoods (with overlaps): {total_nodes_in_small_neighborhoods}")
        print(f"Unique nodes in union of small neighborhoods: {len(all_small_neighborhood_nodes)}")
        print(f"Average overlap: {total_nodes_in_small_neighborhoods / len(small_neighborhoods):.1f} nodes per neighborhood")
    
    def test_basic_functionality(self):
        """Test basic functionality of the hypergraph."""
        # Test that hypergraph has expected attributes
        self.assertIsNotNone(self.hypergraph.x)
        self.assertIsNotNone(self.hypergraph.y)
        self.assertIsNotNone(self.hypergraph.paper_ids)
        self.assertIsNotNone(self.hypergraph.train_mask)
        self.assertIsNotNone(self.hypergraph.val_mask)
        
        # Test that dimensions match
        self.assertEqual(len(self.hypergraph.paper_ids), len(self.hypergraph.x))
        self.assertEqual(len(self.hypergraph.paper_ids), len(self.hypergraph.y))
        self.assertEqual(len(self.hypergraph.paper_ids), len(self.hypergraph.train_mask))
        self.assertEqual(len(self.hypergraph.paper_ids), len(self.hypergraph.val_mask))
        
        # Test that train and val masks are disjoint
        train_val_overlap = torch.logical_and(self.hypergraph.train_mask, self.hypergraph.val_mask)
        self.assertEqual(torch.sum(train_val_overlap).item(), 0, "Train and validation masks should be disjoint")

if __name__ == '__main__':
    unittest.main()