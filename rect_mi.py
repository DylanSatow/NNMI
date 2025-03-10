import torch
class LineMIComputer:
    
    def __init__(self, device='cpu', n_neighbors=3):
        self.device = device
        self.k = n_neighbors

    def compute_counts(self, X,Y, x_radius, y_radius):
        # Do the optimization of this
        # We need sizeof join key * n cause we ened to estimate how many values in the count per join key
        n, m = X.shape
        X1 = X.unsqueeze(1)
        X2 = X.unsqueeze(0)
        Y1 = Y.unsqueeze(1)
        Y2 = Y.unsqueeze(0)
        radius = radius.unsqueeze(1)
        X_diff = torch.abs(X1 - X2)
        Y_diff = torch.abs(Y1 - Y2)
        X_counts = (X_diff < radius).sum(dim=1)
        Y_counts = (Y_diff < radius).sum(dim=1)
        return X_counts - 1, Y_counts - 1

    def compute_radius(self, radius, jk_X_mask, X, Y):
        # Do a pre step where we extract those out, where we construct a new tensor
        # For each join key group, we are looking at a different step which is not monotone
        # What we can do is add a prestep where we first construct
        # Create radius tensor in computemi and then pass it into compute radius and specify which sections of the tensor
        # Bascially just indexing here
        # Maybe use something similar to scatter reduce (bc tensors)
        points = torch.stack((X, Y), dim=2)
        points1 = points.unsqueeze(1)
        points2 = points.unsqueeze(0)
        distances = torch.max(torch.abs(points1 - points2),dim=3)[0].permute(1, 2, 0)
        k_distances, _ = torch.topk(
            distances, k=self.k+1, dim=2, largest=False)
        return k_distances[:, :, -1]
    
    def compute_line_dists(self, X, x_key_inds, Y):
        # O(|J|^2) iterative operation, not very good. Also done iteravely, which I should fix once it works
        J,m = Y.shape
        line_dists = torch.full((J, J), float('inf'), device=self.device)
        line_dists[torch.eye(J, device=self.device, dtype=torch.int)] = 0  # Make diagonals 0
        for jk1 in range(J):
            for jk2 in range(J):
                X1 = X[x_key_inds == jk1]
                X2 = X[x_key_inds == jk2]
                min_dist = torch.min(torch.abs(X1.unsqueeze(0) - X2.unsqueeze(1))).item()
                Y_diff = torch.abs(Y[jk1] - Y[jk2]).item()
                line_dists[jk1, jk2] = max(min_dist, Y_diff)
        return line_dists


    def computeMI(self, X, Y, c_x):
        """
        Compute MI assuming many-to-one join (X is many, Y is one, hence why there is no c_y)
        """

        n = X.shape[0]
        J, m = Y.shape
        x_key_inds = torch.repeat_interleave(torch.arange(J, device=self.device), c_x)

        # Create distance matrix over join keys
        line_dists = self.compute_line_dists(X, x_key_inds, Y)

        # Get Radius
        x_radius = torch.zeros_like(X)
        y_radius = torch.zeros_like(Y)

        for jk in range(J):
            _, line_neighbors = torch.topk(line_dists[jk], min(self.k + 1, J), largest=False)
            
            x_neighborhood_points_mask = torch.isin(x_key_inds,line_neighbors)
            x_neighbors = X[x_neighborhood_points_mask]
            x_curr_mask = (x_key_inds == jk)

            for i in range(n):
                if x_curr_mask[i]:
                    distances = torch.abs(X[i] - x_neighbors)
                    if distances.shape[0] > self.k:
                        x_radius[i] = torch.topk(distances,self.k+1,largest=False)[0][self.k]
                    else:
                        x_radius[i] = distances.max()
            
            y_dists = torch.abs(Y[jk] - Y[line_neighbors])
            y_radius[jk] = y_dists.max()

        nx, ny = self.compute_counts(X, Y, radii)
        # Split compute counts into 2 functions where we take x radius and count of y
        # Other direction is y radius and count of x per join key
        # Then, for each y, we can get a 1d vector where for each point, what is the number of neighboring points per y
        mi = (
            torch.digamma(torch.tensor(X.shape[0], device=self.device))
            + torch.digamma(torch.tensor(self.k, device=self.device))
            - torch.mean(torch.digamma(nx + 1), dim=0)
            - torch.mean(torch.digamma(ny + 1), dim=0)
        )
        return torch.clamp(mi, min=0)
