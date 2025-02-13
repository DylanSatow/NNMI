import torch

class LineMIComputer:
    def __init__(self, device='cpu', n_neighbors=3):
        self.device = torch.device(device)
        self.n_neighbors = n_neighbors

    def compute_line_stats(self, X: torch.Tensor, key_inds: torch.Tensor) -> torch.Tensor:
        num_keys = key_inds.max() + 1
        # For each join key, compute [min_x, max_x]
        x_stats = torch.zeros((num_keys, 2), device=self.device)
        x_stats[:, 0] = torch.scatter_reduce(
            torch.full((num_keys,), float('inf'), device=self.device),
            0, key_inds, X.squeeze(), reduce='amin'
        )
        x_stats[:, 1] = torch.scatter_reduce(
            torch.full((num_keys,), float('-inf'), device=self.device),
            0, key_inds, X.squeeze(), reduce='amax'
        )
        return x_stats

    def compute_line_distances(self, x_stats: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        num_keys = x_stats.shape[0]
        x_stats1 = x_stats.unsqueeze(1)  # (num_keys, 1, 2)
        x_stats2 = x_stats.unsqueeze(0)  # (1, num_keys, 2)
        Y1 = Y.unsqueeze(1)  # (num_keys, 1, m)
        Y2 = Y.unsqueeze(0)  # (1, num_keys, m)

        # Compute line-to-line distances using min/max values
        x_dists = torch.abs(x_stats1 - x_stats2).max(dim=-1)[0]

        # Y distances between aggregated values
        y_dists = torch.abs(Y1 - Y2).max(dim=-1)[0]

        # Combine distances
        distances = torch.max(x_dists, y_dists)

        # Mask out self-distances
        mask = ~torch.eye(num_keys, dtype=torch.bool, device=self.device)
        distances = distances * mask + float('inf') * (~mask)

        return distances

    def estimate_radius_from_k_lines(self, X: torch.Tensor, Y: torch.Tensor,
                                     x_stats: torch.Tensor, key_inds: torch.Tensor,
                                     top_k_indices: torch.Tensor) -> torch.Tensor:
        num_keys = len(top_k_indices)
        radius = torch.zeros((X.shape[0], Y.shape[1]), device=self.device)

        for i in range(num_keys):
            mask_i = (key_inds == i)
            if not mask_i.any():
                continue

            points_i_x = X[mask_i]
            y_i = Y[i]

            # Only examine k-closest lines as specified in 1.2.1
            neighbor_indices = top_k_indices[i]
            neighbor_bounds = x_stats[neighbor_indices]  # (k, 2)
            neighbor_y = Y[neighbor_indices]  # (k, m)

            # For each point in current join key
            for point_idx, x_val in enumerate(points_i_x):
                # Compute distances to k-closest lines
                x_distances = torch.abs(x_val.unsqueeze(-1) - neighbor_bounds).max(dim=-1)[0]

                # For each Y feature
                for feat_idx in range(Y.shape[1]):
                    y_distances = torch.abs(y_i[feat_idx] - neighbor_y[:, feat_idx])
                    feat_distances = torch.max(x_distances, y_distances)

                    # Get kth smallest distance
                    k_smallest = min(self.n_neighbors, len(feat_distances))
                    if k_smallest > 0:
                        values, _ = torch.topk(feat_distances, k=k_smallest, largest=False)
                        point_idx_in_mask = torch.where(mask_i)[0][point_idx]
                        radius[point_idx_in_mask, feat_idx] = values[-1]

        return radius

    def compute_MI(self, X: torch.Tensor, Y: torch.Tensor, c_x: torch.Tensor, c_y: torch.Tensor) -> torch.Tensor:
        # Move inputs to device
        X = X.to(self.device)
        Y = Y.to(self.device)
        c_x = c_x.to(self.device)
        c_y = c_y.to(self.device)

        num_keys = len(c_y)
        x_key_inds = torch.repeat_interleave(torch.arange(num_keys, device=self.device), c_x)

        x_stats = self.compute_line_stats(X, x_key_inds)

        line_distances = self.compute_line_distances(x_stats, Y)
        k = min(self.n_neighbors, num_keys - 1)
        if k < 1:
            k = 1
        _, top_k_indices = torch.topk(
            line_distances, k=k, dim=1, largest=False)

        radius = self.estimate_radius_from_k_lines(X, Y, x_stats, x_key_inds, top_k_indices)

        nx = torch.zeros((X.shape[0], Y.shape[1]), device=self.device)
        ny = torch.zeros((X.shape[0], Y.shape[1]), device=self.device)

        Y_full = Y[x_key_inds]

        for feat_idx in range(Y.shape[1]):
            feat_radius = radius[:, feat_idx:feat_idx+1]

            # Use line bounds to accelerate point counting
            for i in range(num_keys):
                mask_i = (x_key_inds == i)
                if not mask_i.any():
                    continue

                points_i = X[mask_i]
                for point_idx, x_val in enumerate(points_i):
                    point_radius = feat_radius[mask_i][point_idx]

                    # Fast filtering using line bounds
                    potential_matches = (x_val >= (x_stats[:, 0] - point_radius)) & (x_val <= (x_stats[:, 1] + point_radius))

                    if potential_matches.any():
                        point_idx_in_mask = torch.where(mask_i)[0][point_idx]
                        y_val = Y_full[point_idx_in_mask, feat_idx]

                        filtered_keys = torch.where(potential_matches)[0]
                        filtered_y = Y[filtered_keys, feat_idx]

                        # Count points within radius
                        y_dists = torch.abs(y_val - filtered_y)
                        counts = ((y_dists <= point_radius).sum() - 1)

                        nx[point_idx_in_mask, feat_idx] = counts
                        ny[point_idx_in_mask, feat_idx] = counts

        mi = (
            torch.digamma(torch.tensor(X.shape[0], device=self.device, dtype=torch.float)) +
            torch.digamma(torch.tensor(k, device=self.device, dtype=torch.float)) -
            torch.mean(torch.digamma(nx + 1).to(torch.float), dim=0) -
            torch.mean(torch.digamma(ny + 1).to(torch.float), dim=0)
        )

        return torch.clamp(mi, min=0)
