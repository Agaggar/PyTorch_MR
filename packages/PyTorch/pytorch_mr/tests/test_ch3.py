import numpy as np
import torch

import modern_robotics.core as np_mr
import pytorch_mr.core as pmr


def test_screw_to_axis_example():
    q = torch.tensor([3.0, 0.0, 0.0])
    s = torch.tensor([0.0, 0.0, 1.0])
    h = torch.tensor(2.0)
    out = pmr.ScrewToAxis(q, s, h)
    expected = np_mr.ScrewToAxis(q.numpy(), s.numpy(), float(h))
    assert np.allclose(out.cpu().numpy(), expected[None, :], atol=1e-6)


def test_axisang6_example():
    expc6 = torch.tensor([1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    S, theta = pmr.AxisAng6(expc6)
    S_np, theta_np = np_mr.AxisAng6(expc6.numpy())
    assert np.allclose(S.squeeze(0).cpu().numpy(), S_np, atol=1e-6)
    assert np.allclose(theta.squeeze(0).cpu().numpy(), theta_np, atol=1e-6)


def test_project_to_so3_example():
    mat = torch.tensor([[0.675, 0.150, 0.720],
                        [0.370, 0.771, -0.511],
                        [-0.630, 0.619, 0.472]], dtype=torch.float64)
    out = pmr.ProjectToSO3(mat).squeeze(0)
    expected = np_mr.ProjectToSO3(mat.cpu().numpy())
    assert np.allclose(out.cpu().numpy(), expected, atol=1e-5)


def test_project_to_se3_example():
    mat = torch.tensor([[0.675, 0.150, 0.720, 1.2],
                        [0.370, 0.771, -0.511, 5.4],
                        [-0.630, 0.619, 0.472, 3.6],
                        [0.003, 0.002, 0.010, 0.9]], dtype=torch.float64)
    out = pmr.ProjectToSE3(mat).squeeze(0)
    expected = np_mr.ProjectToSE3(mat.cpu().numpy())
    assert np.allclose(out.cpu().numpy(), expected, atol=1e-6)


def test_distance_and_membership_examples():
    so3_bad = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.1, -0.95],
                            [0.0, 1.0, 0.1]])
    dso3 = pmr.DistanceToSO3(so3_bad).squeeze(0).cpu().numpy()
    expected_dso3 = np_mr.DistanceToSO3(so3_bad.cpu().numpy())
    assert np.allclose(dso3, expected_dso3, atol=1e-5)
    assert bool(pmr.TestIfSO3(so3_bad).squeeze(0)) == bool(np_mr.TestIfSO3(so3_bad.cpu().numpy()))

    se3_bad = torch.tensor([[1.0, 0.0, 0.0, 1.2],
                            [0.0, 0.1, -0.95, 1.5],
                            [0.0, 1.0, 0.1, -0.9],
                            [0.0, 0.0, 0.1, 0.98]])
    dse3 = pmr.DistanceToSE3(se3_bad).squeeze(0).cpu().numpy()
    expected_dse3 = np_mr.DistanceToSE3(se3_bad.cpu().numpy())
    assert np.allclose(dse3, expected_dse3, atol=1e-5)
    assert bool(pmr.TestIfSE3(se3_bad).squeeze(0)) == bool(np_mr.TestIfSE3(se3_bad.cpu().numpy()))


def test_broadcast_and_device_basic():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    out = pmr.Normalize(x)
    assert out.shape[-2:] == (3, 1)
    assert out.device == x.device
