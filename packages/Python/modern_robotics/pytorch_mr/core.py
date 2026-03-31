import torch
'''
***************************************************************************
Modern Robotics: Mechanics, Planning, and Control.
Code Library for PyTorch
***************************************************************************
Author: Ayush Gaggar
        Adapted from original Python code by Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes, Kevin Lynch, and Frank Park
Email: agaggar@u.northwestern.edu
Date: July 2025
***************************************************************************
Language: Python
Required library: pytorch
***************************************************************************

This code is a PyTorch implementation of the mathematical operations and transformations in the Modern Robotics package,
and so is best used for *batched* tensors, i.e., tensors with shape (N, ...), where N is the batch size.
For functions after chapter 3, many of the docstrings were written by AI. Code review and shape broadcasting was also completed by AI.
'''

def _as_batch_vec(x: torch.Tensor, dim: int):
    """Ensures vector-like inputs have a batch dimension.

    :param x: Tensor with shape (dim,), (N, dim), or (N, dim, 1)
    :param dim: Expected vector dimension
    :return: Tensor with shape (N, dim)
    """
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 3 and x.shape[-1] == 1 and x.shape[-2] == dim:
        return x.squeeze(-1)
    return x

def _as_batch_mat(x: torch.Tensor, rows: int, cols: int):
    """Ensures matrix-like inputs have a batch dimension.

    :param x: Tensor with shape (rows, cols) or (N, rows, cols)
    :param rows: Number of rows of the matrix
    :param cols: Number of columns of the matrix
    :return: Tensor with shape (N, rows, cols)
    """
    if x.ndim == 2 and x.shape == (rows, cols):
        return x.unsqueeze(0)
    return x

def NearZero(val: torch.Tensor, eps=1e-6):
    """Check if input tensor values are close to zero."""
    return torch.abs(val) < eps

def Normalize(v: torch.Tensor):
    """Normalizes a vector to unit length.

    :param v: A vector, shape (N, 3, 1), (N, 3), (3,)
    :return: A normalized vector, shape (N, 3)
    """
    if v.ndim == 1:
        v = v.unsqueeze(0)
    if v.ndim == 2:
        v = v.unsqueeze(-1)
    norm = torch.linalg.norm(v, dim=-2, keepdim=True)
    return v / norm.clamp(min=1e-6)  # Avoid division by zero

def RotInv(R: torch.Tensor):
    """Inverts a rotation matrix.

    :param R: A rotation matrix, shape (N, 3, 3)
    :return: The inverse of R, which is the transpose of R, shape (N, 3, 3)
    """
    if R.ndim == 2:
        R = R.unsqueeze(0)
    return torch.transpose(R, -2, -1)

def VecToso3(omega: torch.Tensor):
    """Converts a 3-vector to an so(3) representation

    :param omg: angular velocity vector, shape (N, 3)
    :return: The skew symmetric representation of omg, i.e., an so(3) matrix,
             shape (N, 3, 3)
    """
    if omega.ndim > 2:
        omega = omega.reshape(-1, 3)
    elif omega.ndim == 1:
        omega = omega.unsqueeze(0)
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    row0 = torch.stack([torch.zeros_like(wx), -wz, wy], dim=1)
    row1 = torch.stack([wz, torch.zeros_like(wy), -wx], dim=1)
    row2 = torch.stack([-wy, wx, torch.zeros_like(wz)], dim=1)
    so3 = torch.stack([row0, row1, row2], dim=1)
    return so3

def so3ToVec(so3mat: torch.Tensor):
    """Converts an so(3) representation to a 3-vector

    :param so3mat: A skew-symmetric matrix, shape (N, 3, 3)
    :return: The 3-vector corresponding to so3mat, shape (N, 3, 1)
    """
    if so3mat.ndim == 2:
        so3mat = so3mat.unsqueeze(0)
    return torch.stack([
        so3mat[:, 2, 1],
        so3mat[:, 0, 2],
        so3mat[:, 1, 0]
    ], dim=-1).unsqueeze(-1)


def AxisAng3(expc3: torch.Tensor):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation, shape (N, 3, 1)
    :return omghat: A unit rotation axis, shape (N, 3)
    :return theta: The corresponding rotation angle, shape (N,)
    """
    expc3 = _as_batch_vec(expc3, 3)
    if expc3.ndim == 2:
        expc3 = expc3.unsqueeze(-1)  # (N,3,1)
    theta = torch.linalg.norm(expc3, dim=-2).reshape(expc3.shape[0])
    omega_hat = torch.zeros_like(expc3)
    nonzero = ~NearZero(theta)
    omega_hat[nonzero] = expc3[nonzero] / theta[nonzero][:, None, None]
    return omega_hat, theta

def MatrixExp3(so3mat: torch.Tensor):
    """Computes the matrix exponential of a matrix in so(3)

    :param so3mat: A skew-symmetric matrix, shape (N, 3, 3)
    :return: The matrix exponential of so3mat, shape (N, 3, 3)

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    so3mat = _as_batch_mat(so3mat, 3, 3)
    omgtheta = so3ToVec(so3mat)
    nonzero = ~NearZero(torch.linalg.norm(omgtheta, dim=-2).reshape(so3mat.shape[0]))
    add_me = torch.zeros_like(so3mat)
    theta_nonzero = AxisAng3(omgtheta)[1][nonzero][:, None, None]
    omgmat_nonzero = so3mat[nonzero] / theta_nonzero
    add_me[nonzero] = torch.sin(theta_nonzero) * omgmat_nonzero + (1 - torch.cos(theta_nonzero)) * torch.bmm(omgmat_nonzero, omgmat_nonzero)
    identity = torch.eye(3, device=so3mat.device, dtype=so3mat.dtype)
    identity = identity.repeat(so3mat.shape[0], 1, 1)
    return identity + add_me

def MatrixLog3(R: torch.Tensor):
    """Computes the matrix logarithm of rotation matrix

    :param R: Nx3x3 rotation matrix, shape (N, 3, 3)
    :return: The matrix logarithm of R, shape (N, 3, 3)
    """
    R = _as_batch_mat(R, 3, 3)
    #TODO: broadcast instead of iterating
    trace_R = R.diagonal(offset=0, dim1=1, dim2=2).sum(-1)  # (N,)
    acosinput = torch.clamp((trace_R - 1) / 2.0, -1.0, 1.0)
    logm = torch.zeros_like(R, device=R.device, dtype=R.dtype)
    for idx, rot in enumerate(R):
        if torch.abs(1 - acosinput[idx]) <= 1e-6:
            continue
        elif torch.abs(acosinput[idx] + 1) <= 1e-6:
            if not NearZero(1 + rot[2][2]):
                omg = (1.0 / torch.sqrt(torch.clip(2 * (1 + rot[2][2]), min=1e-6))) \
                      * torch.stack([rot[0][2], rot[1][2], 1 + rot[2][2]], dim=0)
            elif not NearZero(1 + rot[1][1]):
                omg = (1.0 / torch.sqrt(torch.clip(2 * (1 + rot[1][1]), min=1e-6))) \
                      * torch.stack([rot[0][1], 1 + rot[1][1], rot[2][1]], dim=0)
            else:
                omg = (1.0 / torch.sqrt(torch.clip(2 * (1 + rot[0][0]), min=1e-6))) \
                      * torch.stack([1 + rot[0][0], rot[1][0], rot[2][0]], dim=0)
            logm[idx] = VecToso3(torch.pi * omg)
        else:
            theta = torch.acos(acosinput[idx])
            logm[idx] = theta / (2.0 * torch.sin(theta)) * (rot - torch.transpose(rot, -2, -1))
    if torch.any(torch.isnan(logm)):
        raise ValueError("NaN values in MatrixLog3")
    return logm

def RpToTrans(R: torch.Tensor, p: torch.Tensor):
    """Converts rotation matrices and position vectors into homogeneous
    transformation matrices

    :param R: Rotation matrix, (N, 3, 3)
    :param p: A 3-vector, (N, 3)
    :return: A homogeneous transformation matrix corresponding to the inputs
    """
    if p.ndim == 2:
        p = p.unsqueeze(-1)
    bottom_row = torch.tensor([0, 0, 0, 1], device=R.device, dtype=R.dtype)
    bottom_row = bottom_row.expand(R.size(0), 1, 4)
    return torch.cat([torch.cat([R, p], dim=-1), 
                      bottom_row], dim=-2).float().to(device=R.device)

def TransToRp(T: torch.Tensor):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector, given T is (N, 4, 4).

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix, (N, 3, 3)
    :return p: The corresponding position vector, (N, 3, 1)
    """
    if T.ndim == 2:
        T = T.unsqueeze(0)
    return T[:, 0: 3, 0: 3], T[:, 0: 3, 3].unsqueeze(-1)

def TransInv(T: torch.Tensor):
    """Inverts a homogeneous transformation matrix, given T is (N, 4, 4).

    :param T: A homogeneous transformation matrix, (N, 4, 4)
    :return: The inverse of T, (N, 4, 4)
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    """
    R, p = TransToRp(T)
    Rt = torch.transpose(R, -2, -1)
    if p.ndim == 2:
        p = p.unsqueeze(-1)
    bottom_row = torch.tensor([0, 0, 0, 1], device=R.device, dtype=R.dtype)
    bottom_row = bottom_row.expand(R.size(0), 1, 4)
    return torch.cat([torch.cat([Rt, -torch.bmm(Rt, p)], dim=-1), 
                      bottom_row], dim=-2)
    
def VecTose3(V: torch.Tensor):
    """Converts a spatial velocity vector into a 4x4 matrix in se3

    :param V: A 6-vector representing a spatial velocity, shape (N, 6)
    :return: The 4x4 se3 representation of V, shape (N, 4, 4)
    """
    omega = V[:, :3]
    v = V[:, 3:]
    if v.ndim == 2:
        v = v.unsqueeze(-1)
        omega = omega.unsqueeze(-1)
    # bottom_row = torch.zeros((V.shape[0], 1, 4), device=V.device, dtype=V.dtype)
    # bottom_row[:, 0, 3] = 1.
    bottom_row = torch.zeros((V.size(0), 1, 4), device=V.device, dtype=V.dtype)
    return torch.cat([torch.cat([VecToso3(omega), v], dim=-1), 
                      bottom_row], dim=-2)

def se3ToVec(se3mat: torch.Tensor):
    """ Converts se3 matrices into spatial velocity vector

    :param se3mat: SE3 Matrix, shape (N, 4, 4)
    :return: The spatial velocity 6-vector corresponding to se3mat, shape (N, 6, 1)
    """
    if se3mat.ndim == 2:
        se3mat = se3mat.unsqueeze(0)
    vecm = torch.stack([
        se3mat[:, 2, 1],
        se3mat[:, 0, 2],
        se3mat[:, 1, 0],
        se3mat[:, 0, 3],
        se3mat[:, 1, 3],
        se3mat[:, 2, 3],
    ], dim=1)
    return vecm.unsqueeze(-1)

def Adjoint(T: torch.Tensor):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix, shape (N, 4, 4)
    :return: The adjoint representation of T, shape (N, 6, 6)
    """
    R, p = TransToRp(T)
    top_left = R
    top_right = torch.zeros((R.shape[0], 3, 3), device=T.device, dtype=T.dtype)
    bottom_left = torch.bmm(VecToso3(p), R)
    bottom_right = R
    return torch.cat([torch.cat([top_left, top_right], dim=-1),
                      torch.cat([bottom_left, bottom_right], dim=-1)], dim=-2)

def ScrewToAxis(q: torch.Tensor, s: torch.Tensor, h: torch.Tensor):
    """Takes a parametric description of a screw axis and converts it to a
    normalized screw axis.

    :param q: A point on the screw axis, shape (N, 3)
    :param s: The direction of the screw axis, shape (N, 3)
    :param h: The pitch of the screw, shape (N,)
    :return: The axis representation of the screw, shape (N, 6)
    """
    q = _as_batch_vec(q, 3)
    s = _as_batch_vec(s, 3)
    if h.ndim == 0:
        h = h.unsqueeze(0)
    if h.ndim == 2 and h.shape[-1] == 1:
        h = h.squeeze(-1)
    if h.ndim == 1 and h.shape[0] == 1 and q.shape[0] > 1:
        h = h.expand(q.shape[0])
    return torch.cat([s, torch.cross(q, s, dim=-1) + h.unsqueeze(-1) * s], dim=-1)

def AxisAng6(expc6: torch.Tensor):
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form

    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta, shape (N, 6) or (N, 6, 1)
    :return S: The corresponding normalized screw axis, shape (N, 6)
    :return theta: The distance traveled along/about S, shape (N, 1) or (N,)
    """
    expc6 = _as_batch_vec(expc6, 6)
    theta = torch.linalg.norm(expc6[:, :3], dim=-1)
    mask = NearZero(theta)
    theta[mask] = torch.linalg.norm(expc6[mask, 3:], dim=-1)
    theta_safe = torch.where(NearZero(theta), torch.ones_like(theta), theta)
    return expc6 / theta_safe.unsqueeze(-1), theta

def MatrixExp6(se3mat: torch.Tensor):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates

    :param se3mat: Matrix in se3, shape (N, 4, 4)
    :return: Matrix exponential of se3mat, shape (N, 4, 4)
    """
    omgtheta = so3ToVec(se3mat[:, :3, :3])
    nonzero = ~NearZero(torch.linalg.norm(omgtheta, dim=-2).reshape(se3mat.shape[0]))
    
    first_part = MatrixExp3(se3mat[:, :3, :3])
    second_part = se3mat[:, :3, 3].unsqueeze(-1).clone()
    identity = torch.eye(3, device=se3mat.device, dtype=se3mat.dtype)
    identity = identity.repeat(se3mat.shape[0], 1, 1)
    bottom_row = torch.tensor([0, 0, 0, 1], device=se3mat.device, dtype=se3mat.dtype)
    bottom_row = bottom_row.expand(se3mat.size(0), 1, 4)  # (N, 1, 4)

    theta_nonzero = AxisAng3(omgtheta)[1][nonzero][:, None, None]
    omgmat_nonzero = se3mat[:, :3, :3][nonzero] / theta_nonzero
    second_part[nonzero] = torch.bmm(identity[nonzero] * theta_nonzero \
                                        + (1 - torch.cos(theta_nonzero)) * omgmat_nonzero \
                                        + (theta_nonzero - torch.sin(theta_nonzero)) \
                                        * torch.bmm(omgmat_nonzero, omgmat_nonzero),
                                     second_part[nonzero] / theta_nonzero)
    omgmat_nonzero = se3mat[:, :3, :3][nonzero] / theta_nonzero
    
    return torch.cat([torch.cat([first_part, second_part], dim=-1), 
                      bottom_row], dim=-2)

def MatrixLog6(T: torch.Tensor):
    """Computes the matrix logarithm of a homogeneous transformation matrix

    :param T: Matrix in SE3, shape (N, 4, 4)
    :return: Matrix logarithm of T, shape (N, 4, 4)
    """
    T = _as_batch_mat(T, 4, 4)
    rot, second = TransToRp(T)
    all_rots = MatrixLog3(rot)
    if second.ndim == 2:
        second = second.unsqueeze(-1)

    first = torch.zeros_like(rot, device=T.device, dtype=T.dtype)
    bottom_row = torch.zeros((T.shape[0], 1, 4), device=T.device, dtype=T.dtype)
    logm = torch.zeros_like(T, device=T.device, dtype=T.dtype)

    #TODO: broadcast instead of iterating
    nonzero = ~(NearZero(all_rots, eps=1e-4).sum(dim=2).sum(dim=1) == 9)
    logm[~nonzero] = torch.cat([torch.cat([first[~nonzero], second[~nonzero]], dim=-1),
                                 bottom_row[~nonzero]], dim=-2)
    trace_R = rot.diagonal(offset=0, dim1=1, dim2=2).sum(-1)  # (N,)
    if torch.any(torch.isnan(trace_R)):
        raise ValueError("NaN values in trace_R")
    theta_nonzero = torch.acos(torch.clip((trace_R - 1) / 2.0, -1., 1.))[nonzero]

    identity = torch.eye(3, device=T.device, dtype=T.dtype)
    identity = identity.repeat(T.shape[0], 1, 1)
    logm[nonzero] = torch.cat([torch.cat([all_rots[nonzero], 
                                          (identity[nonzero] - all_rots[nonzero] / 2.0 \
                                                + torch.clamp((1.0 / theta_nonzero - 1.0 / torch.tan(theta_nonzero / 2.0) / 2.), min=-1e3, max=1e4)[:, None, None] \
                                                * torch.bmm(all_rots[nonzero], all_rots[nonzero]) / theta_nonzero[:, None, None]) @ second[nonzero]], dim=-1),
                                 bottom_row[nonzero]], dim=-2)
    if torch.any(torch.isnan(logm)):
        raise ValueError("NaN values in MatrixLog6")
    return logm

def ProjectToSO3(R: torch.Tensor):
    """Projects a rotation matrix onto the nearest SO(3) matrix

    :param R: A rotation matrix, shape (N, 3, 3)
    :return: The projected rotation matrix, shape (N, 3, 3)
    """
    R = _as_batch_mat(R, 3, 3)
    U, _, Vh = torch.linalg.svd(R)
    Rproj = torch.matmul(U, Vh)
    det = torch.linalg.det(Rproj)
    fix_mask = det < 0
    if torch.any(fix_mask):
        Rproj_fix = Rproj[fix_mask].clone()
        Rproj_fix[:, :, 2] = -Rproj_fix[:, :, 2]
        Rproj[fix_mask] = Rproj_fix
    return Rproj

def ProjectToSE3(T: torch.Tensor):
    """Returns a projection of mat into SE(3)

    :param mat: A 4x4 matrix to project to SE(3)
    :return: The closest matrix to T that is in SE(3)
    Projects a matrix mat to the closest matrix in SE(3) using singular-value
    decomposition.
    """
    T = _as_batch_mat(T, 4, 4)
    out = T.clone()
    out[:, :3, :3] = ProjectToSO3(T[:, :3, :3])
    out[:, 3, :] = torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).expand(T.shape[0], 4)
    return out

def DistanceToSO3(R: torch.Tensor):
    """Returns the Frobenius norm to describe the distance of mat from the
    SO(3) manifold

    :param R: A matrix, shape (N, 3, 3)
    :return: A quantity describing the distance of mat from the SO(3) manifold, shape (N,)
    """
    R = _as_batch_mat(R, 3, 3)
    det = torch.linalg.det(R)
    out = torch.full((R.shape[0],), 1e9, device=R.device, dtype=R.dtype)
    mask = det > 0
    if torch.any(mask):
        RtR = torch.matmul(torch.transpose(R[mask], -2, -1), R[mask])
        I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(RtR.shape[0], 3, 3)
        out[mask] = torch.linalg.norm(RtR - I, dim=(1, 2))
    return out

def DistanceToSE3(T: torch.Tensor):
    """Returns the Frobenius norm to describe the distance of mat from the
    SE(3) manifold

    :param T: A matrix, shape (N, 4, 4)
    :return: A quantity describing the distance of mat from the SE(3) manifold, shape (N,)

    Computes the distance from mat to the SE(3) manifold using the following
    method:
    Compute the determinant of matR, the top 3x3 submatrix of mat.
    If det(matR) <= 0, return a large number.
    If det(matR) > 0, replace the top 3x3 submatrix of mat with matR^T.matR,
    and set the first three entries of the fourth column of mat to zero. Then
    return norm(mat - I).

    Example Input:
        T = torch.tensor([[ 1.0,  0.0,   0.0,   1.2 ],
                        [ 0.0,  0.1,  -0.95,  1.5 ],
                        [ 0.0,  1.0,   0.1,  -0.9 ],
                        [ 0.0,  0.0,   0.1,   0.98 ]], dtype=torch.float64)
    Output:
        0.134931
    """
    T = _as_batch_mat(T, 4, 4)
    matR = T[:, :3, :3]
    det = torch.linalg.det(matR)
    out = torch.full((T.shape[0],), 1e9, device=T.device, dtype=T.dtype)
    mask = det > 0
    if torch.any(mask):
        RtR = torch.matmul(torch.transpose(matR[mask], -2, -1), matR[mask])
        top = torch.cat([RtR, torch.zeros((RtR.shape[0], 3, 1), device=T.device, dtype=T.dtype)], dim=-1)
        bottom = T[mask, 3:4, :]
        mat = torch.cat([top, bottom], dim=-2)
        I = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mat.shape[0], 4, 4)
        out[mask] = torch.linalg.norm(mat - I, dim=(1, 2))
    return out

def TestIfSO3(R: torch.Tensor):
    """Returns true if mat is close to or on the manifold SO(3)

    :param mat: A 3x3 matrix
    :return: True if mat is very close to or in SO(3), false otherwise
    """
    return torch.abs(DistanceToSO3(R)) < 1e-3

def TestIfSE3(T: torch.Tensor):
    """Returns true if mat is close to or on the manifold SE(3)

    :param mat: A 4x4 matrix
    :return: True if mat is very close to or in SE(3), false otherwise
    """
    return torch.abs(DistanceToSE3(T)) < 1e-3

def FKinBody(M: torch.Tensor, Blist: torch.Tensor, thetalist: torch.Tensor):
    """Computes forward kinematics in the body frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-effector
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, with axes as columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-effector
             frame when the joints are at the specified coordinates (Body frame)
    """
    M = _as_batch_mat(M, 4, 4)
    Blist = Blist if Blist.ndim == 3 else Blist.unsqueeze(0)
    thetalist = _as_batch_vec(thetalist, Blist.shape[-1])
    T = M.clone()
    n = thetalist.shape[-1]
    for i in range(n):
        T = torch.bmm(T, MatrixExp6(VecTose3(Blist[:, :, i] * thetalist[:, i:i+1])))
    return T

def FKinSpace(M: torch.Tensor, Slist: torch.Tensor, thetalist: torch.Tensor):
    """Computes forward kinematics in the space frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-effector
    :param Slist: The joint screw axes in the space frame when the manipulator
                  is at the home position, with axes as columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-effector
             frame when the joints are at the specified coordinates (Space frame)
    """
    M = _as_batch_mat(M, 4, 4)
    Slist = Slist if Slist.ndim == 3 else Slist.unsqueeze(0)
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1])
    T = M.clone()
    n = thetalist.shape[-1]
    for i in range(n - 1, -1, -1):
        T = torch.bmm(MatrixExp6(VecTose3(Slist[:, :, i] * thetalist[:, i:i+1])), T)
    return T

def JacobianBody(Blist: torch.Tensor, thetalist: torch.Tensor):
    """Computes the body Jacobian for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, with axes as columns
    :param thetalist: A list of joint coordinates
    :return: The body Jacobian corresponding to the inputs
    """
    Blist = Blist if Blist.ndim == 3 else Blist.unsqueeze(0)
    thetalist = _as_batch_vec(thetalist, Blist.shape[-1])
    Jb = Blist.clone().to(dtype=torch.float32 if Blist.dtype in (torch.int32, torch.int64) else Blist.dtype)
    T = torch.eye(4, device=Blist.device, dtype=Jb.dtype).unsqueeze(0).repeat(Blist.shape[0], 1, 1)
    for i in range(thetalist.shape[-1] - 2, -1, -1):
        T = torch.bmm(T, MatrixExp6(VecTose3(Blist[:, :, i + 1] * (-thetalist[:, i + 1:i + 2]))))
        Jb[:, :, i] = torch.bmm(Adjoint(T), Blist[:, :, i].unsqueeze(-1)).squeeze(-1)
    return Jb

def JacobianSpace(Slist: torch.Tensor, thetalist: torch.Tensor):
    """Computes the space Jacobian for an open chain robot

    :param Slist: The joint screw axes in the space frame when the manipulator
                  is at the home position, with axes as columns
    :param thetalist: A list of joint coordinates
    :return: The space Jacobian corresponding to the inputs
    """
    Slist = Slist if Slist.ndim == 3 else Slist.unsqueeze(0)
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1])
    Js = Slist.clone().to(dtype=torch.float32 if Slist.dtype in (torch.int32, torch.int64) else Slist.dtype)
    T = torch.eye(4, device=Slist.device, dtype=Js.dtype).unsqueeze(0).repeat(Slist.shape[0], 1, 1)
    for i in range(1, thetalist.shape[-1]):
        T = torch.bmm(T, MatrixExp6(VecTose3(Slist[:, :, i - 1] * thetalist[:, i - 1:i])))
        Js[:, :, i] = torch.bmm(Adjoint(T), Slist[:, :, i].unsqueeze(-1)).squeeze(-1)
    return Js

def IKinBody(Blist: torch.Tensor, M: torch.Tensor, T: torch.Tensor, thetalist0: torch.Tensor, eomg=1e-6, ev=1e-6):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, with axes as columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles
    :param eomg: Tolerance on orientation error
    :param ev: Tolerance on linear position error
    :return thetalist: Joint angles that achieve T within the specified tolerances
    :return success: True if a solution is found, False otherwise
    Uses an iterative Newton-Raphson root-finding method with maxiterations=20.
    """
    Blist = Blist if Blist.ndim == 3 else Blist.unsqueeze(0)
    M = _as_batch_mat(M, 4, 4)
    T = _as_batch_mat(T, 4, 4)
    thetalist = _as_batch_vec(thetalist0, Blist.shape[-1]).clone()
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(torch.bmm(TransInv(FKinBody(M, Blist, thetalist)), T))).squeeze(-1)
    err = (torch.linalg.norm(Vb[:, :3], dim=-1) > eomg) | (torch.linalg.norm(Vb[:, 3:], dim=-1) > ev)
    while torch.any(err) and i < maxiterations:
        Jb = JacobianBody(Blist, thetalist)
        for b in range(thetalist.shape[0]):
            if err[b]:
                thetalist[b] = thetalist[b] + torch.matmul(torch.linalg.pinv(Jb[b]), Vb[b])
        i = i + 1
        Vb = se3ToVec(MatrixLog6(torch.bmm(TransInv(FKinBody(M, Blist, thetalist)), T))).squeeze(-1)
        err = (torch.linalg.norm(Vb[:, :3], dim=-1) > eomg) | (torch.linalg.norm(Vb[:, 3:], dim=-1) > ev)
    return thetalist, ~err

def IKinSpace(Slist: torch.Tensor, M: torch.Tensor, T: torch.Tensor, thetalist0: torch.Tensor, eomg=1e-6, ev=1e-6):
    """Computes inverse kinematics in the space frame for an open chain robot

    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, with axes as columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles
    :param eomg: Tolerance on orientation error
    :param ev: Tolerance on linear position error
    :return thetalist: Joint angles that achieve T within the specified tolerances
    :return success: True if a solution is found, False otherwise
    Uses an iterative Newton-Raphson root-finding method with maxiterations=20.
    """
    Slist = Slist if Slist.ndim == 3 else Slist.unsqueeze(0)
    M = _as_batch_mat(M, 4, 4)
    T = _as_batch_mat(T, 4, 4)
    thetalist = _as_batch_vec(thetalist0, Slist.shape[-1]).clone()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = torch.bmm(Adjoint(Tsb), se3ToVec(MatrixLog6(torch.bmm(TransInv(Tsb), T)))).squeeze(-1)
    err = (torch.linalg.norm(Vs[:, :3], dim=-1) > eomg) | (torch.linalg.norm(Vs[:, 3:], dim=-1) > ev)
    while torch.any(err) and i < maxiterations:
        Js = JacobianSpace(Slist, thetalist)
        for b in range(thetalist.shape[0]):
            if err[b]:
                thetalist[b] = thetalist[b] + torch.matmul(torch.linalg.pinv(Js[b]), Vs[b])
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = torch.bmm(Adjoint(Tsb), se3ToVec(MatrixLog6(torch.bmm(TransInv(Tsb), T)))).squeeze(-1)
        err = (torch.linalg.norm(Vs[:, :3], dim=-1) > eomg) | (torch.linalg.norm(Vs[:, 3:], dim=-1) > ev)
    return thetalist, ~err

def ad(V: torch.Tensor):
    """Computes the adjoint representation of a spatial velocity vector

    :param V: A spatial velocity vector, shape (N, 6)
    :return: The adjoint representation of V, shape (N, 6, 6)
    """
    V = _as_batch_vec(V, 6)
    omgmat = VecToso3(V[:, :3])
    zeros = torch.zeros((V.shape[0], 3, 3), device=V.device, dtype=V.dtype)
    return torch.cat([torch.cat([omgmat, zeros], dim=-1),
                      torch.cat([VecToso3(V[:, 3:]), omgmat], dim=-1)], dim=-2)

def InverseDynamics(thetalist: torch.Tensor, dthetalist: torch.Tensor, ddthetalist: torch.Tensor, g: torch.Tensor, Ftip: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor):
    """Computes inverse dynamics in the space frame for an open chain robot

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame {n+1}
    :param Mlist: List of link frames {i} relative to {i-1} at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, with axes as columns
    :return: The n-vector of required joint forces/torques
    This function uses forward-backward Newton-Euler iterations to solve:
    taulist = M(thetalist)ddthetalist + c(thetalist,dthetalist)
              + g(thetalist) + Jtr(thetalist)Ftip
    """
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1] if Slist.ndim > 1 else thetalist.shape[-1])
    dthetalist = _as_batch_vec(dthetalist, thetalist.shape[-1])
    ddthetalist = _as_batch_vec(ddthetalist, thetalist.shape[-1])
    if g.ndim == 1:
        g = g.unsqueeze(0).expand(thetalist.shape[0], 3)
    Ftip = _as_batch_vec(Ftip, 6)
    Mlist = Mlist if Mlist.ndim == 4 else Mlist.unsqueeze(0)
    Glist = Glist if Glist.ndim == 4 else Glist.unsqueeze(0)
    Slist = Slist if Slist.ndim == 3 else Slist.unsqueeze(0)
    n = thetalist.shape[-1]
    B = thetalist.shape[0]
    taulist = torch.zeros((B, n), device=thetalist.device, dtype=thetalist.dtype)
    for b in range(B):
        Mi = torch.eye(4, device=thetalist.device, dtype=thetalist.dtype)
        Ai = torch.zeros((6, n), device=thetalist.device, dtype=thetalist.dtype)
        AdTi = [None] * (n + 1)
        Vi = torch.zeros((6, n + 1), device=thetalist.device, dtype=thetalist.dtype)
        Vdi = torch.zeros((6, n + 1), device=thetalist.device, dtype=thetalist.dtype)
        Vdi[:, 0] = torch.cat([torch.zeros(3, device=thetalist.device, dtype=thetalist.dtype), -g[b]])
        AdTi[n] = Adjoint(TransInv(Mlist[b, n].unsqueeze(0))).squeeze(0)
        Fi = Ftip[b].clone()
        for i in range(n):
            Mi = torch.matmul(Mi, Mlist[b, i])
            Ai[:, i] = torch.matmul(Adjoint(TransInv(Mi.unsqueeze(0))).squeeze(0), Slist[b, :, i])
            AdTi[i] = Adjoint(torch.matmul(MatrixExp6(VecTose3((Ai[:, i] * (-thetalist[b, i])).unsqueeze(0))).squeeze(0),
                                          TransInv(Mlist[b, i].unsqueeze(0)).squeeze(0)).unsqueeze(0)).squeeze(0)
            Vi[:, i + 1] = torch.matmul(AdTi[i], Vi[:, i]) + Ai[:, i] * dthetalist[b, i]
            Vdi[:, i + 1] = torch.matmul(AdTi[i], Vdi[:, i]) + Ai[:, i] * ddthetalist[b, i] + torch.matmul(ad(Vi[:, i + 1].unsqueeze(0)).squeeze(0), Ai[:, i]) * dthetalist[b, i]
        for i in range(n - 1, -1, -1):
            Fi = torch.matmul(torch.transpose(AdTi[i + 1], -2, -1), Fi) + torch.matmul(Glist[b, i], Vdi[:, i + 1]) - torch.matmul(torch.transpose(ad(Vi[:, i + 1].unsqueeze(0)).squeeze(0), -2, -1), torch.matmul(Glist[b, i], Vi[:, i + 1]))
            taulist[b, i] = torch.dot(Fi, Ai[:, i])
    return taulist

def MassMatrix(thetalist: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor):
    """Computes the mass matrix of an open chain robot based on configuration

    :param thetalist: A list of joint variables
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, with axes as columns
    :return: The numerical inertia matrix M(thetalist)
    """
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1] if Slist.ndim > 1 else thetalist.shape[-1])
    n = thetalist.shape[-1]
    M = torch.zeros((thetalist.shape[0], n, n), device=thetalist.device, dtype=thetalist.dtype)
    for i in range(n):
        dd = torch.zeros_like(thetalist)
        dd[:, i] = 1
        M[:, :, i] = InverseDynamics(thetalist, torch.zeros_like(thetalist), dd, torch.zeros((thetalist.shape[0], 3), device=thetalist.device, dtype=thetalist.dtype), torch.zeros((thetalist.shape[0], 6), device=thetalist.device, dtype=thetalist.dtype), Mlist, Glist, Slist)
    return M

def VelQuadraticForces(thetalist: torch.Tensor, dthetalist: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor):
    """Computes Coriolis and centripetal terms in inverse dynamics.

    :param thetalist: Joint variables
    :param dthetalist: Joint rates
    :param Mlist: List of link frames
    :param Glist: Spatial inertia matrices
    :param Slist: Joint screw axes
    :return: c(thetalist, dthetalist)
    """
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1] if Slist.ndim > 1 else thetalist.shape[-1])
    dthetalist = _as_batch_vec(dthetalist, thetalist.shape[-1])
    return InverseDynamics(thetalist, dthetalist, torch.zeros_like(thetalist), torch.zeros((thetalist.shape[0], 3), device=thetalist.device, dtype=thetalist.dtype), torch.zeros((thetalist.shape[0], 6), device=thetalist.device, dtype=thetalist.dtype), Mlist, Glist, Slist)

def GravityForces(thetalist: torch.Tensor, g: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor):
    """Computes joint forces/torques required to overcome gravity.

    :param thetalist: Joint variables
    :param g: Gravity vector
    :param Mlist: List of link frames
    :param Glist: Spatial inertia matrices
    :param Slist: Joint screw axes
    :return: Gravity torque vector
    """
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1] if Slist.ndim > 1 else thetalist.shape[-1])
    if g.ndim == 1:
        g = g.unsqueeze(0).expand(thetalist.shape[0], 3)
    return InverseDynamics(thetalist, torch.zeros_like(thetalist), torch.zeros_like(thetalist), g, torch.zeros((thetalist.shape[0], 6), device=thetalist.device, dtype=thetalist.dtype), Mlist, Glist, Slist)

def EndEffectorForces(thetalist: torch.Tensor, Ftip: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor):
    """Computes joint forces/torques required only to create end-effector force.

    :param thetalist: Joint variables
    :param Ftip: End-effector wrench in frame {n+1}
    :param Mlist: List of link frames
    :param Glist: Spatial inertia matrices
    :param Slist: Joint screw axes
    :return: Joint forces/torques due to Ftip
    """
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1] if Slist.ndim > 1 else thetalist.shape[-1])
    Ftip = _as_batch_vec(Ftip, 6)
    return InverseDynamics(thetalist, torch.zeros_like(thetalist), torch.zeros_like(thetalist), torch.zeros((thetalist.shape[0], 3), device=thetalist.device, dtype=thetalist.dtype), Ftip, Mlist, Glist, Slist)

def ForwardDynamics(thetalist: torch.Tensor, dthetalist: torch.Tensor, taulist: torch.Tensor, g: torch.Tensor, Ftip: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor):
    """Computes forward dynamics in the space frame for an open chain robot.

    :param thetalist: Joint variables
    :param dthetalist: Joint rates
    :param taulist: Joint forces/torques
    :param g: Gravity vector
    :param Ftip: End-effector wrench
    :param Mlist: List of link frames
    :param Glist: Spatial inertia matrices
    :param Slist: Joint screw axes
    :return: Joint accelerations
    """
    thetalist = _as_batch_vec(thetalist, Slist.shape[-1] if Slist.ndim > 1 else thetalist.shape[-1])
    dthetalist = _as_batch_vec(dthetalist, thetalist.shape[-1])
    taulist = _as_batch_vec(taulist, thetalist.shape[-1])
    if g.ndim == 1:
        g = g.unsqueeze(0).expand(thetalist.shape[0], 3)
    Ftip = _as_batch_vec(Ftip, 6)
    M = MassMatrix(thetalist, Mlist, Glist, Slist)
    rhs = taulist - VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist) - GravityForces(thetalist, g, Mlist, Glist, Slist) - EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)
    return torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

def EulerStep(thetalist: torch.Tensor, dthetalist: torch.Tensor, ddthetalist: torch.Tensor, dt: float):
    """Performs a single Euler step for the given joint angles, velocities, and accelerations.

    :param thetalist: Joint angles, shape (N, 6)
    :param dthetalist: Joint velocities, shape (N, 6)
    :param ddthetalist: Joint accelerations, shape (N, 6)
    :param dt: Time step for the Euler integration
    :return: Updated joint angles and velocities after the Euler step
    """
    return thetalist + dthetalist * dt, dthetalist + ddthetalist * dt

def InverseDynamicsTrajectory(thetamat: torch.Tensor, dthetamatrix: torch.Tensor, ddthetamatrix: torch.Tensor, g: torch.Tensor, Ftipmatrix: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor):
    """Calculates required joint torques along a trajectory using inverse dynamics.

    :param thetamat: N x n matrix of joint variables
    :param dthetamatrix: N x n matrix of joint velocities
    :param ddthetamatrix: N x n matrix of joint accelerations
    :param g: Gravity vector
    :param Ftipmatrix: N x 6 matrix of end-effector wrenches
    :param Mlist: List of link frames
    :param Glist: Spatial inertia matrices
    :param Slist: Joint screw axes
    :return: N x n matrix of joint torques
    """
    thetamat = _as_batch_mat(thetamat, thetamat.shape[-2], thetamat.shape[-1]) if thetamat.ndim == 2 else thetamat
    if thetamat.ndim == 2:
        thetamat = thetamat.unsqueeze(0)
    if dthetamatrix.ndim == 2:
        dthetamatrix = dthetamatrix.unsqueeze(0)
    if ddthetamatrix.ndim == 2:
        ddthetamatrix = ddthetamatrix.unsqueeze(0)
    if Ftipmatrix.ndim == 2:
        Ftipmatrix = Ftipmatrix.unsqueeze(0)
    B, N, _ = thetamat.shape
    out = torch.zeros_like(thetamat)
    for b in range(B):
        for i in range(N):
            out[b, i] = InverseDynamics(thetamat[b, i], dthetamatrix[b, i], ddthetamatrix[b, i], g[b] if g.ndim == 2 else g, Ftipmatrix[b, i], Mlist[b] if Mlist.ndim == 4 else Mlist, Glist[b] if Glist.ndim == 4 else Glist, Slist[b] if Slist.ndim == 3 else Slist).squeeze(0)
    return out

def ForwardDynamicsTrajectory(thetamat: torch.Tensor, dthetamatrix: torch.Tensor, taumat: torch.Tensor, g: torch.Tensor, Ftipmatrix: torch.Tensor, Mlist: torch.Tensor, Glist: torch.Tensor, Slist: torch.Tensor, dt: float, intRes: int):
    """Simulates motion given an open-loop history of joint torques.

    :param thetamat: Initial joint variables
    :param dthetamatrix: Initial joint rates
    :param taumat: N x n matrix of torques
    :param g: Gravity vector
    :param Ftipmatrix: N x 6 matrix of end-effector wrenches
    :param Mlist: List of link frames
    :param Glist: Spatial inertia matrices
    :param Slist: Joint screw axes
    :param dt: Time step between consecutive torque commands
    :param intRes: Euler integration resolution per step
    :return thetamat: N x n matrix of resulting joint angles
    :return dthetamat: N x n matrix of resulting joint velocities
    """
    if thetamat.ndim == 1:
        thetamat = thetamat.unsqueeze(0)
    if dthetamatrix.ndim == 1:
        dthetamatrix = dthetamatrix.unsqueeze(0)
    if taumat.ndim == 2:
        taumat = taumat.unsqueeze(0)
    if Ftipmatrix.ndim == 2:
        Ftipmatrix = Ftipmatrix.unsqueeze(0)
    B, N, n = taumat.shape
    theta_out = torch.zeros((B, N, n), device=taumat.device, dtype=taumat.dtype)
    dtheta_out = torch.zeros((B, N, n), device=taumat.device, dtype=taumat.dtype)
    thetalist = thetamat.clone()
    dthetalist = dthetamatrix.clone()
    theta_out[:, 0] = thetalist
    dtheta_out[:, 0] = dthetalist
    for i in range(N - 1):
        for _ in range(intRes):
            dd = ForwardDynamics(thetalist, dthetalist, taumat[:, i], g, Ftipmatrix[:, i], Mlist, Glist, Slist)
            thetalist, dthetalist = EulerStep(thetalist, dthetalist, dd, 1.0 * dt / intRes)
        theta_out[:, i + 1] = thetalist
        dtheta_out[:, i + 1] = dthetalist
    return theta_out, dtheta_out

def CubicTimeScaling(Tf, t):
    """Computes s(t) for a cubic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a third-order
             polynomial motion that begins and ends at zero velocity

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.216
    """
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

def QuinticTimeScaling(Tf, t):
    """Computes s(t) for a quintic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a fifth-order
             polynomial motion that begins and ends at zero velocity and zero
             acceleration

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.16308
    """
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
           + 6 * (1.0 * t / Tf) ** 5

def JointTrajectory(thetalist: torch.Tensor, thetaend: torch.Tensor, Tf: float, N: int, method):
    """Computes a straight-line trajectory in joint space.

    :param thetastart: Initial joint variables
    :param thetaend: Final joint variables
    :param Tf: Total time from rest to rest
    :param N: Number of points in the trajectory
    :param method: Time scaling method (3 for cubic, 5 for quintic)
    :return: An N x n trajectory of joint variables
    """
    thetalist = _as_batch_vec(thetalist, thetalist.shape[-1] if thetalist.ndim > 1 else thetalist.shape[0])
    thetaend = _as_batch_vec(thetaend, thetalist.shape[-1])
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = torch.zeros((thetalist.shape[0], N, thetalist.shape[-1]), device=thetalist.device, dtype=thetalist.dtype)
    for i in range(N):
        s = CubicTimeScaling(Tf, timegap * i) if method == 3 else QuinticTimeScaling(Tf, timegap * i)
        traj[:, i] = s * thetaend + (1 - s) * thetalist
    return traj

def ScrewTrajectory(Xstart: torch.Tensor, Xend: torch.Tensor, Tf: float, N: int, method):
    """Computes a trajectory as a list of N SE(3) matrices following screw motion.

    :param Xstart: Initial end-effector configuration
    :param Xend: Final end-effector configuration
    :param Tf: Total motion time
    :param N: Number of points
    :param method: Time scaling method (3 or 5)
    :return: N SE(3) matrices, separated by Tf/(N-1)
    """
    Xstart = _as_batch_mat(Xstart, 4, 4)
    Xend = _as_batch_mat(Xend, 4, 4)
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = torch.zeros((Xstart.shape[0], N, 4, 4), device=Xstart.device, dtype=Xstart.dtype)
    for i in range(N):
        s = CubicTimeScaling(Tf, timegap * i) if method == 3 else QuinticTimeScaling(Tf, timegap * i)
        traj[:, i] = torch.bmm(Xstart, MatrixExp6(MatrixLog6(torch.bmm(TransInv(Xstart), Xend)) * s))
    return traj

def CartesianTrajectory(Xstart: torch.Tensor, Xend: torch.Tensor, Tf: float, N: int, method):
    """Computes a trajectory with straight-line translation and decoupled rotation.

    :param Xstart: Initial end-effector configuration
    :param Xend: Final end-effector configuration
    :param Tf: Total motion time
    :param N: Number of points
    :param method: Time scaling method (3 or 5)
    :return: N SE(3) matrices, separated by Tf/(N-1)
    """
    Xstart = _as_batch_mat(Xstart, 4, 4)
    Xend = _as_batch_mat(Xend, 4, 4)
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = torch.zeros((Xstart.shape[0], N, 4, 4), device=Xstart.device, dtype=Xstart.dtype)
    Rstart, pstart = TransToRp(Xstart)
    Rend, pend = TransToRp(Xend)
    for i in range(N):
        s = CubicTimeScaling(Tf, timegap * i) if method == 3 else QuinticTimeScaling(Tf, timegap * i)
        R = torch.bmm(Rstart, MatrixExp3(MatrixLog3(torch.bmm(torch.transpose(Rstart, -2, -1), Rend)) * s))
        p = s * pend + (1 - s) * pstart
        traj[:, i] = RpToTrans(R, p.squeeze(-1))
    return traj

def ComputedTorque(**kwargs):
    """Computes joint control torques at a particular time instant.

    Keyword args follow the NumPy implementation:
    thetalist, dthetalist, eint, g, Mlist, Glist, Slist,
    thetalistd, dthetalistd, ddthetalistd, Kp, Ki, Kd.
    """
    thetalist = kwargs["thetalist"]
    dthetalist = kwargs["dthetalist"]
    eint = kwargs["eint"]
    g = kwargs["g"]
    Mlist = kwargs["Mlist"]
    Glist = kwargs["Glist"]
    Slist = kwargs["Slist"]
    thetalistd = kwargs["thetalistd"]
    dthetalistd = kwargs["dthetalistd"]
    ddthetalistd = kwargs["ddthetalistd"]
    Kp = kwargs["Kp"]
    Ki = kwargs["Ki"]
    Kd = kwargs["Kd"]
    thetalist = _as_batch_vec(thetalist, thetalist.shape[-1] if thetalist.ndim > 1 else thetalist.shape[0])
    dthetalist = _as_batch_vec(dthetalist, thetalist.shape[-1])
    eint = _as_batch_vec(eint, thetalist.shape[-1])
    thetalistd = _as_batch_vec(thetalistd, thetalist.shape[-1])
    dthetalistd = _as_batch_vec(dthetalistd, thetalist.shape[-1])
    ddthetalistd = _as_batch_vec(ddthetalistd, thetalist.shape[-1])
    e = thetalistd - thetalist
    return torch.bmm(MassMatrix(thetalist, Mlist, Glist, Slist), (Kp * e + Ki * (eint + e) + Kd * (dthetalistd - dthetalist)).unsqueeze(-1)).squeeze(-1) + InverseDynamics(thetalist, dthetalist, ddthetalistd, g, torch.zeros((thetalist.shape[0], 6), device=thetalist.device, dtype=thetalist.dtype), Mlist, Glist, Slist)

def SimulateControl(**kwargs):
    """Simulates computed torque control over a desired trajectory.

    Keyword args follow the NumPy implementation:
    thetalist, dthetalist, g, Ftipmat, Mlist, Glist, Slist,
    thetamatd, dthetamatd, ddthetamatd, gtilde, Mtildelist, Gtildelist,
    Kp, Ki, Kd, dt, intRes.
    :return taumat: Commanded joint torques over time
    :return thetamat: Actual joint angles over time
    """
    thetalist = kwargs["thetalist"]
    dthetalist = kwargs["dthetalist"]
    g = kwargs["g"]
    Ftipmat = kwargs["Ftipmat"]
    Mlist = kwargs["Mlist"]
    Glist = kwargs["Glist"]
    Slist = kwargs["Slist"]
    thetamatd = kwargs["thetamatd"]
    dthetamatd = kwargs["dthetamatd"]
    ddthetamatd = kwargs["ddthetamatd"]
    gtilde = kwargs["gtilde"]
    Mtildelist = kwargs["Mtildelist"]
    Gtildelist = kwargs["Gtildelist"]
    Kp = kwargs["Kp"]
    Ki = kwargs["Ki"]
    Kd = kwargs["Kd"]
    dt = kwargs["dt"]
    intRes = kwargs["intRes"]
    if thetalist.ndim == 1:
        thetalist = thetalist.unsqueeze(0)
    if dthetalist.ndim == 1:
        dthetalist = dthetalist.unsqueeze(0)
    if thetamatd.ndim == 2:
        thetamatd = thetamatd.unsqueeze(0)
    if dthetamatd.ndim == 2:
        dthetamatd = dthetamatd.unsqueeze(0)
    if ddthetamatd.ndim == 2:
        ddthetamatd = ddthetamatd.unsqueeze(0)
    if Ftipmat.ndim == 2:
        Ftipmat = Ftipmat.unsqueeze(0)
    B, N, n = thetamatd.shape
    thetacurrent = thetalist.clone()
    dthetacurrent = dthetalist.clone()
    eint = torch.zeros((B, n), device=thetamatd.device, dtype=thetamatd.dtype)
    taumat = torch.zeros((B, N, n), device=thetamatd.device, dtype=thetamatd.dtype)
    thetamat = torch.zeros((B, N, n), device=thetamatd.device, dtype=thetamatd.dtype)
    for i in range(N):
        taulist = ComputedTorque(
            thetalist=thetacurrent, dthetalist=dthetacurrent, eint=eint, g=gtilde,
            Mlist=Mtildelist, Glist=Gtildelist, Slist=Slist, thetalistd=thetamatd[:, i],
            dthetalistd=dthetamatd[:, i], ddthetalistd=ddthetamatd[:, i], Kp=Kp, Ki=Ki, Kd=Kd
        )
        for _ in range(intRes):
            dd = ForwardDynamics(thetacurrent, dthetacurrent, taulist, g, Ftipmat[:, i], Mlist, Glist, Slist)
            thetacurrent, dthetacurrent = EulerStep(thetacurrent, dthetacurrent, dd, 1.0 * dt / intRes)
        taumat[:, i] = taulist
        thetamat[:, i] = thetacurrent
        eint = eint + dt * (thetamatd[:, i] - thetacurrent)
    return taumat, thetamat
