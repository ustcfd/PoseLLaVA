import numpy as np
import torch

def compute_similarity_transform(source_points, target_points):
    """Computes a similarity transform (sR, t) that takes a set of 3D points
    source_points (N x 3) closest to a set of 3D points target_points, where R
    is an 3x3 rotation matrix, t 3x1 translation, s scale.

    And return the
    transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
    Procrutes problem.
    Notes:
        Points number: N
    Args:
        source_points (np.ndarray([N, 3])): Source point set.
        target_points (np.ndarray([N, 3])): Target point set.
    Returns:
        source_points_hat (np.ndarray([N, 3])): Transformed source point set.
    """

    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3

    source_points = source_points.T
    target_points = target_points.T

    # 1. Remove mean.
    mu1 = source_points.mean(axis=1, keepdims=True)
    mu2 = target_points.mean(axis=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform the source points:
    source_points_hat = scale * R.dot(source_points) + t

    source_points_hat = source_points_hat.T

    return source_points_hat


def batch_compute_similarity_transform_torch(S1, S2, return_transform=False):

    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    S1 shape: [B,3,N], S2 shape: [B, 3, N]
    '''
    input_dtype = S1.dtype 
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])
    
    if input_dtype != torch.float32:
        S1 = S1.to(torch.float32)
        S2 = S2.to(torch.float32)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))
    
    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)
    if return_transform:
        return S1_hat.to(input_dtype), (scale, R, t)
    else:
        return S1_hat.to(input_dtype)

def compute_mpjpe(preds, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape,print(preds.shape, target.shape)  # BxJx3
    return torch.mean(torch.norm(preds - target, p=2, dim=-1))

def compute_pa_mpjpe(pred, target):
    batch_size = pred.shape[0]
    pred_tranformed = batch_compute_similarity_transform_torch(pred.view(batch_size, -1, 3), target.view(batch_size, -1, 3))
    pa_mpjpe = compute_mpjpe(pred_tranformed,target.view(batch_size, -1, 3))
    return pa_mpjpe