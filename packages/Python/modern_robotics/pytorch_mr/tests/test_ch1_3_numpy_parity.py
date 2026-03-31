import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "shape",
    [
        (3,),       # (dim,)
        (1, 3),     # (N, dim)
        (5, 3),     # (N, dim)
        (5, 3, 1),  # (N, dim, 1)
    ],
)
def test_normalize_numpy_parity(pt_impl, np_ref, shape):
    x = torch.randn(*shape, dtype=torch.float64)
    y = pt_impl.Normalize(x)

    # NumPy Normalize expects a vector, not batched; apply per-batch
    if x.ndim == 1:
        y_np = np_ref.Normalize(x.detach().cpu().numpy())
        assert y.shape == (1, 3, 1)  # current PyTorch Normalize contract
        assert np.allclose(y.squeeze(0).squeeze(-1).cpu().numpy(), y_np, atol=1e-6)
    else:
        batch = x.shape[0]
        y_np = []
        for i in range(batch):
            xi = x[i]
            if xi.ndim == 2 and xi.shape[-1] == 1:
                xi = xi.squeeze(-1)
            y_np.append(np_ref.Normalize(xi.detach().cpu().numpy()))
        y_np = np.stack(y_np, axis=0)
        assert np.allclose(y.squeeze(-1).cpu().numpy(), y_np, atol=1e-6)


@pytest.mark.parametrize("val", [-1e-7, 0.0, 1e-5, 1e-4])
def test_nearzero_numpy_parity(pt_impl, np_ref, val):
    t = torch.tensor(val, dtype=torch.float64)
    out = bool(pt_impl.NearZero(t))
    expected = bool(np_ref.NearZero(float(val)))
    assert out == expected


def test_rotinv_numpy_parity(pt_impl, np_ref):
    R = torch.tensor([[0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=torch.float64)
    out = pt_impl.RotInv(R).squeeze(0).cpu().numpy()
    expected = np_ref.RotInv(R.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize(
    "omega_shape",
    [(3,), (1, 3), (7, 3), (7, 3, 1)],
)
def test_vectoso3_and_back_numpy_parity(pt_impl, np_ref, omega_shape):
    omega = torch.randn(*omega_shape, dtype=torch.float64)
    so3 = pt_impl.VecToso3(omega)

    if omega.ndim == 1:
        expected = np_ref.VecToso3(omega.cpu().numpy())
        assert np.allclose(so3.squeeze(0).cpu().numpy(), expected, atol=1e-12)
        back = pt_impl.so3ToVec(so3).squeeze(0).squeeze(-1).cpu().numpy()
        back_expected = np_ref.so3ToVec(expected)
        assert np.allclose(back, back_expected, atol=1e-12)
    else:
        batch = omega.shape[0]
        expected = []
        for i in range(batch):
            oi = omega[i]
            if oi.ndim == 2 and oi.shape[-1] == 1:
                oi = oi.squeeze(-1)
            expected.append(np_ref.VecToso3(oi.cpu().numpy()))
        expected = np.stack(expected, axis=0)
        assert np.allclose(so3.cpu().numpy(), expected, atol=1e-12)


def test_axisang3_numpy_parity(pt_impl, np_ref):
    expc3 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    omghat_t, theta_t = pt_impl.AxisAng3(expc3)

    omghat_np, theta_np = np_ref.AxisAng3(expc3.cpu().numpy())
    assert np.allclose(omghat_t.squeeze(0).squeeze(-1).cpu().numpy(), omghat_np, atol=1e-6)
    assert np.allclose(theta_t.squeeze(0).cpu().numpy(), theta_np, atol=1e-6)


def test_matrixexp3_numpy_example_parity(pt_impl, np_ref):
    so3 = torch.tensor([[0.0, -3.0, 2.0],
                        [3.0, 0.0, -1.0],
                        [-2.0, 1.0, 0.0]], dtype=torch.float64)
    out = pt_impl.MatrixExp3(so3).squeeze(0).cpu().numpy()
    expected = np_ref.MatrixExp3(so3.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-6)


def test_matrixlog3_numpy_example_parity(pt_impl, np_ref):
    R = torch.tensor([[0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=torch.float64)
    out = pt_impl.MatrixLog3(R).squeeze(0).cpu().numpy()
    expected = np_ref.MatrixLog3(R.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-6)


def test_rp_trans_roundtrip_numpy_parity(pt_impl, np_ref):
    R = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
    Tt = pt_impl.RpToTrans(R.unsqueeze(0), p.unsqueeze(0)).squeeze(0).cpu().numpy()
    Tn = np_ref.RpToTrans(R.cpu().numpy(), p.cpu().numpy())
    assert np.allclose(Tt, Tn, atol=1e-12)

    Rt, pt = pt_impl.TransToRp(torch.tensor(Tn, dtype=torch.float64))  # type: ignore
    Rt = Rt.squeeze(0).cpu().numpy()
    pt = pt.squeeze(0).squeeze(-1).cpu().numpy()
    Rn, pn = np_ref.TransToRp(Tn)
    assert np.allclose(Rt, Rn, atol=1e-12)
    assert np.allclose(pt, pn, atol=1e-12)


def test_transinv_numpy_example_parity(pt_impl, np_ref):
    T = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0],
                      [0.0, 1.0, 0.0, 3.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    out = pt_impl.TransInv(T).squeeze(0).cpu().numpy()
    expected = np_ref.TransInv(T.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-12)


def test_vectose3_and_back_numpy_parity(pt_impl, np_ref):
    V = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    se3 = pt_impl.VecTose3(V.unsqueeze(0)).squeeze(0).cpu().numpy()
    expected = np_ref.VecTose3(V.cpu().numpy())
    assert np.allclose(se3, expected, atol=1e-12)

    back = pt_impl.se3ToVec(torch.tensor(expected, dtype=torch.float64)).squeeze(0).squeeze(-1).cpu().numpy()
    back_expected = np_ref.se3ToVec(expected)
    assert np.allclose(back, back_expected, atol=1e-12)


def test_adjoint_numpy_example_parity(pt_impl, np_ref):
    T = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0],
                      [0.0, 1.0, 0.0, 3.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    out = pt_impl.Adjoint(T).squeeze(0).cpu().numpy()
    expected = np_ref.Adjoint(T.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-12)


def test_matrixexp6_and_log6_numpy_examples_parity(pt_impl, np_ref):
    se3 = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.57079632, 2.35619449],
                        [0.0, 1.57079632, 0.0, 2.35619449],
                        [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    out = pt_impl.MatrixExp6(se3.unsqueeze(0)).squeeze(0).cpu().numpy()
    expected = np_ref.MatrixExp6(se3.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-6)

    T = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0],
                      [0.0, 1.0, 0.0, 3.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    out_log = pt_impl.MatrixLog6(T).squeeze(0).cpu().numpy()
    expected_log = np_ref.MatrixLog6(T.cpu().numpy())
    assert np.allclose(out_log, expected_log, atol=1e-6)

