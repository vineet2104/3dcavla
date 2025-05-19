import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class STN3d(nn.Module):
    '''
    Computes a 3x3 transformation matrix for input point clouds to align them in a canonical space,
    ensuring that the PointNet is invariant to input transformations (e.g. rotation and scaling)
    '''
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    '''
    Generalizes the STN3d module to handle input feature spaces of arbitrary dimension k.
    '''
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetProjector(nn.Module):
    def __init__(self, input_dim: int, llm_dim: int):
        """
        PointNet Projector for projecting features into LLM-compatible space.
        Args:
            input_dim (int): Dimension of input features (e.g., 1024 from PointNet).
            llm_dim (int): Target embedding dimension for LLM.
        """
        super().__init__()

        intermediate_dim = 2 * input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim, bias=True),
            nn.GELU(),
            nn.Linear(intermediate_dim, llm_dim, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to project features into LLM-compatible space.
        Args:
            features (torch.Tensor): Input tensor of shape [batch_size, feature_dim].
        
        Returns:
            torch.Tensor: Projected features of shape [batch_size, llm_dim].
        """
        projected_features = self.proj(features)  
        return projected_features

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, use_MLP = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.use_MLP = use_MLP
        self.proj = nn.Linear(1024, 4096) # project dimention of global features to match LLM dimension
        # self.proj = PointNetProjector(1024, 4096)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            if self.use_MLP:
                x = self.proj(x)
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def check_weights_equality(state_dict, loaded_state_dict, verbose=True, rtol=1e-5, atol=1e-8):
    """
    Check if two state dictionaries have similar weights within tolerance.
    
    Args:
        state_dict: First state dict (saved weights)
        loaded_state_dict: Second state dict (loaded model weights)
        verbose: Whether to print detailed information about differences
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
    
    Returns:
        bool: Whether the weights are equal within tolerance
    """
    are_weights_equal = True
    mismatched_shapes = []
    mismatched_values = []
    missing_in_loaded = []
    extra_in_loaded = []

    # Check saved weights are in loaded weights
    for key in state_dict.keys():
        if key not in loaded_state_dict:
            missing_in_loaded.append(key)
            are_weights_equal = False
            continue

        # Move tensors to CPU and convert to float32 for comparison
        saved_tensor = state_dict[key].cpu().float()
        loaded_tensor = loaded_state_dict[key].cpu().float()

        if saved_tensor.shape != loaded_tensor.shape:
            mismatched_shapes.append((
                key,
                saved_tensor.shape,
                loaded_tensor.shape
            ))
            are_weights_equal = False
            continue

        # Use allclose instead of exact equality for better numerical comparison
        if not torch.allclose(saved_tensor, loaded_tensor, rtol=rtol, atol=atol):
            # Calculate difference statistics
            diff = (saved_tensor - loaded_tensor)
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            rel_diff = torch.max(torch.abs(diff / (torch.abs(saved_tensor) + 1e-8))).item()

            mismatched_values.append((
                key,
                max_diff,
                mean_diff,
                rel_diff
            ))
            are_weights_equal = False

    # Check for extra keys
    for key in loaded_state_dict.keys():
        if key not in state_dict:
            extra_in_loaded.append(key)
            are_weights_equal = False

    if verbose:
        # Print report
        if missing_in_loaded:
            print("\nKeys missing in loaded model:")
            for key in missing_in_loaded:
                print(f"  - {key}")

        if extra_in_loaded:
            print("\nExtra keys in loaded model:")
            for key in extra_in_loaded:
                print(f"  - {key}")

        if mismatched_shapes:
            print("\nShape mismatches:")
            for key, shape1, shape2 in mismatched_shapes:
                print(f"  - {key}: {shape1} vs {shape2}")

        if mismatched_values:
            print("\nValue mismatches:")
            for key, max_diff, mean_diff, rel_diff in mismatched_values:
                print(f"  - {key}:")
                print(f"    Max absolute diff: {max_diff:.6f}")
                print(f"    Mean absolute diff: {mean_diff:.6f}")
                print(f"    Max relative diff: {rel_diff:.6f}")
        
        if are_weights_equal:
            print("All weights are exactly the same!")
        else:
            print("Summary of differences:")
            print(f"  Missing keys: {len(missing_in_loaded)}")
            print(f"  Extra keys: {len(extra_in_loaded)}")
            print(f"  Shape mismatches: {len(mismatched_shapes)}")
            print(f"  Value mismatches: {len(mismatched_values)}")

    return are_weights_equal