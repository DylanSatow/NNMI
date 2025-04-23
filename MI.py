from abc import ABC, abstractmethod
import torch
import numpy as np

class MIEstimator(ABC):
    """Abstract base class for mutual information estimators for BCD."""

    def __init__(self, device: str = 'cpu'):
        self.is_fitted = False
        self.device = device

    @staticmethod
    def _assign_bins(x, min_val, width):
        bin_indices = ((x - min_val) // width).to(torch.int64)
        return bin_indices

    def _entropy_from_inds(
        self, hist_size, ind_unique, ind_inverse, counts, pad_inds, save=False
    ):
        hist = torch.zeros((hist_size, ind_inverse.shape[1]),
                           device=self.device, dtype=torch.float)
        hist.scatter_add_(0, ind_inverse, counts)
        # filter_ = torch.arange(hist.shape[0], device=self.device)
        # mask = filter_[:, None] > pad_inds[None, :]
        if save:
            import pickle
            n = int(torch.sum(hist[:, 0]).item())
            itm = eps + hist / torch.sum(hist[:, 0])
            ent = -torch.sum(itm * torch.log(itm), dim=0)
            with open(f'hist_{n}.pkl', 'wb') as file:
                pickle.dump((hist, ent), file)
        hist = eps + hist / torch.sum(hist[:, 0])
        # hist[mask] = 1
        return -torch.sum(hist * torch.log(hist), dim=0)

    @abstractmethod
    def fit(self, be, coeff_xy, coeff_yx):
        """
        Estimate the mutual information between X and Y.

        Returns:
            float: Estimated mutual information value
        """

    @abstractmethod
    def compute_mi_diff(self) -> float:
        """
        Estimate the mutual information between X and Y.

        Returns:
            float: Estimated mutual information value
        """
        if not self.is_fitted:
            raise ValueError(
                "Estimator must be fitted before calling compute_mi()")
        
class NNMIEstimator(MIEstimator):
    def __init__(self, device='cpu', n_neighbors=3):
        self.k = n_neighbors
        super().__init__(device)

    def _get_radius(self,n, all_results):
        radius = np.zeros((n,))
        
        a = 0
        for jk, dists in all_results.items():
            dists_tensor = torch.from_numpy(dists)
            kth_values = torch.topk(dists_tensor, k=self.k, dim=1, largest=False)[0][:,-1]
            
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
    
    def fit(self, X, Y, c_x):
        self.n = X.shape[0]
        all_results = {} # TODO
        self.radius = self._get_radius(self.n, all_results)
        self.n_x = self._get_counts(self.radius, X)
        self.n_y = self._get_counts(self.radius, Y)
        self.is_fitted = True
        
    def compute_mi_diff(self):
        super().compute_mi_diff()

        mi = (
            torch.digamma(torch.tensor(self.n, device=self.device))
            + torch.digamma(torch.tensor(self.k, device=self.device))
            - torch.mean(torch.digamma(self.n_x + 1), dim=0)
            - torch.mean(torch.digamma(self.n_y + 1), dim=0)
        )
        return torch.clamp(mi, min=0)

            
