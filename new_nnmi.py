import torch
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

eps = 1e-5

class MIEstimator(ABC):

    def __init__(self, device: str = 'cpu'):
        self.is_fitted = False
        self.device = device

    @abstractmethod
    def fit(self, X, Y, c_x, c_y, coeff):
        pass

    @abstractmethod
    def compute_mi(self) -> float:
        if not self.is_fitted:
            raise ValueError(
                "Estimator must be fitted before calling compute_mi()")


class NewNNMI(MIEstimator):

    def __init__(self, device: str = 'cpu', n_neighbors: int = 3):
        super().__init__(device)
        self.k = n_neighbors

    @staticmethod
    def _get_join_rows(X, c_x, c_y):
        bounds = torch.cumsum(c_x, 0)
        starts = torch.cat([torch.tensor([0]), bounds[:-1]])

        indices = torch.cat(
            [torch.arange(start, end).repeat(rep) for start, end, rep in zip(
                starts, bounds, c_y
            )])
        return torch.index_select(X, 0, indices)

    def fit(self, X, Y, c_x, c_y, coeff):
        deg = coeff.shape[0] - 1
        X_join = self._get_join_rows(X, c_x, c_y)
        Y_join = torch.repeat_interleave(
            Y, torch.repeat_interleave(c_x, c_y), dim=0)

        X_powers = torch.stack([X_join ** i for i in range(deg + 1)], dim=-1)
        pred = torch.sum(X_powers * coeff.t().unsqueeze(0), dim=2)
        self.exp = X_join
        self.res = Y_join - pred
        self.is_fitted = True
        return self

    def _get_radius(self, n, all_results, k):
        radius = np.zeros((n,))
        
        a = 0
        for jk, dists in all_results.items():
            dists_tensor = torch.from_numpy(dists)
            kth_values = torch.topk(dists_tensor, k=k, dim=1, largest=False)[0][:,-1]
            
            radius[a:a+len(kth_values)] = kth_values.numpy()
            a += len(kth_values)
            
        return radius

    def _get_counts(self, radius, X):
        lower = X - radius
        upper = X + radius
        
        X_sorted = np.sort(X)
        lower_ind = np.searchsorted(X_sorted, lower)
        upper_ind = np.searchsorted(X_sorted, upper)
        return upper_ind - lower_ind + 1

    def _compute_all_distances(self, X, Y):
        # Fix this
        # Initialize dictionaries to store results
        all_results = {}
        nn_jk_inds = {}
        
        # Convert to numpy arrays if they're tensors
        X_np = X.numpy() if torch.is_tensor(X) else X
        Y_np = Y.numpy() if torch.is_tensor(Y) else Y
        
        # Stack X and Y for distance calculation
        points = np.column_stack((X_np, Y_np))
        
        # Calculate distances using batch_find_k_closest
        # For simplicity, we'll use a single key 'all' as in the original implementation
        key = 'all'
        target_key = 0  # Assuming we're only dealing with one target key
        
        # Calculate distances using the batch_find_k_closest method
        # This is a placeholder - you'll need to implement the actual batch_find_k_closest method
        distances = self.batch_find_k_closest(points, points, exclude_self=True)
        
        # Store the results
        nn_jk_inds[key] = np.full(distances.shape[1], target_key)
        all_results[key] = distances
        
        print('------------------------------Micro Benchmark------------------------------')
        # Count non-NaN values
        total_non_nan = 0
        for key, distances in all_results.items():
            non_nan_count = np.count_nonzero(~np.isnan(distances))
            total_non_nan += non_nan_count
        
        print(f"Total non-NaN values: {total_non_nan}")
        
        min_vals, max_vals = float('inf'), float('-inf')
        for _, jks in nn_jk_inds.items():
            if len(jks) < min_vals:
                min_vals = len(jks)
            elif len(jks) > max_vals:
                max_vals = len(jks)
        
        print(f"Min cols in extended treat: {min_vals}, Max: {max_vals}")
        print('------------------------------Micro Benchmark------------------------------')
        
        return all_results, nn_jk_inds
        
    def batch_find_k_closest(self, sorted_array, values, exclude_self=False, threshold=None):

        # This is a placeholder implementation
        # You'll need to implement the actual algorithm for finding k closest points
        
        # For now, we'll use a simple approach based on the original implementation
        n = len(values)
        k = self.k
        
        # Initialize result array
        result = np.zeros((n, k))
        
        # For each point in values, find k closest points in sorted_array
        for i in range(n):
            # Calculate distances to all points in sorted_array
            distances = np.max(np.abs(values[i] - sorted_array), axis=1)
            
            # Exclude self if requested
            if exclude_self:
                distances[i] = np.inf
            
            # Apply threshold if provided
            if threshold is not None:
                distances[distances > threshold] = np.inf
            
            # Find k closest points
            closest_indices = np.argsort(distances)[:k]
            result[i] = distances[closest_indices]
        
        return result

    def compute_mi(self) -> float:
        super().compute_mi()
        
        X_np = self.exp.numpy()
        Y_np = self.res.numpy()
        
        all_results, _ = self._compute_all_distances(self.exp, self.res)
        
        radius = self._get_radius(X_np.shape[0], all_results, self.k)
        
        nx = self._get_counts(radius, X_np)
        ny = self._get_counts(radius, Y_np)
        
        nx_tensor = torch.from_numpy(nx).to(self.device)
        ny_tensor = torch.from_numpy(ny).to(self.device)
        
        mi = (
            torch.digamma(torch.tensor(X_np.shape[0], device=self.device))
            + torch.digamma(torch.tensor(self.k, device=self.device))
            - torch.mean(torch.digamma(nx_tensor + 1), dim=0)
            - torch.mean(torch.digamma(ny_tensor + 1), dim=0)
        )
        
        return torch.clamp(mi, min=0).item() 