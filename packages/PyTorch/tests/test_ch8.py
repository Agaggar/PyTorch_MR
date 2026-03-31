import numpy as np
import torch

import modern_robotics.core as np_mr
import pytorch_mr.core as pmr


def robot3():
    M01 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.089159],
                        [0.0, 0.0, 0.0, 1.0]])
    M12 = torch.tensor([[0.0, 0.0, 1.0, 0.28],
                        [0.0, 1.0, 0.0, 0.13585],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
    M23 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, -0.1197],
                        [0.0, 0.0, 1.0, 0.395],
                        [0.0, 0.0, 0.0, 1.0]])
    M34 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.14225],
                        [0.0, 0.0, 0.0, 1.0]])
    G1 = torch.diag(torch.tensor([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7]))
    G2 = torch.diag(torch.tensor([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393]))
    G3 = torch.diag(torch.tensor([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275]))
    Mlist = torch.stack([M01, M12, M23, M34], dim=0)
    Glist = torch.stack([G1, G2, G3], dim=0)
    Slist = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, -0.089, 0.0, 0.0],
                          [0.0, 1.0, 0.0, -0.089, 0.0, 0.425]]).T
    return Mlist, Glist, Slist


def test_ad_example():
    V = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = pmr.ad(V).squeeze(0)
    expected = torch.tensor([[0.0, -3.0, 2.0, 0.0, 0.0, 0.0],
                             [3.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                             [-2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, -6.0, 5.0, 0.0, -3.0, 2.0],
                             [6.0, 0.0, -4.0, 3.0, 0.0, -1.0],
                             [-5.0, 4.0, 0.0, -2.0, 1.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-6)
    expected_np = np_mr.ad(V.cpu().numpy())
    assert np.allclose(out.cpu().numpy(), expected_np, atol=1e-6)


def test_inverse_dynamics_and_related_examples():
    Mlist, Glist, Slist = robot3()
    thetalist = torch.tensor([0.1, 0.1, 0.1])
    dthetalist = torch.tensor([0.1, 0.2, 0.3])
    ddthetalist = torch.tensor([2.0, 1.5, 1.0])
    g = torch.tensor([0.0, 0.0, -9.8])
    Ftip = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    tau = pmr.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist).squeeze(0)
    expected_tau = np_mr.InverseDynamics(
        thetalist.cpu().numpy(), dthetalist.cpu().numpy(), ddthetalist.cpu().numpy(),
        g.cpu().numpy(), Ftip.cpu().numpy(), Mlist.cpu().numpy(), Glist.cpu().numpy(), Slist.cpu().numpy()
    )
    assert np.allclose(tau.cpu().numpy(), expected_tau, atol=1e-3)

    M = pmr.MassMatrix(thetalist, Mlist, Glist, Slist).squeeze(0)
    expected_M = np_mr.MassMatrix(thetalist.cpu().numpy(), Mlist.cpu().numpy(), Glist.cpu().numpy(), Slist.cpu().numpy())
    assert np.allclose(M.cpu().numpy(), expected_M, atol=1e-3)

    c = pmr.VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist).squeeze(0)
    expected_c = np_mr.VelQuadraticForces(thetalist.cpu().numpy(), dthetalist.cpu().numpy(), Mlist.cpu().numpy(), Glist.cpu().numpy(), Slist.cpu().numpy())
    assert np.allclose(c.cpu().numpy(), expected_c, atol=1e-4)

    grav = pmr.GravityForces(thetalist, g, Mlist, Glist, Slist).squeeze(0)
    expected_g = np_mr.GravityForces(thetalist.cpu().numpy(), g.cpu().numpy(), Mlist.cpu().numpy(), Glist.cpu().numpy(), Slist.cpu().numpy())
    assert np.allclose(grav.cpu().numpy(), expected_g, atol=1e-3)

    tip = pmr.EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist).squeeze(0)
    expected_tip = np_mr.EndEffectorForces(thetalist.cpu().numpy(), Ftip.cpu().numpy(), Mlist.cpu().numpy(), Glist.cpu().numpy(), Slist.cpu().numpy())
    assert np.allclose(tip.cpu().numpy(), expected_tip, atol=1e-3)


def test_forward_dynamics_and_trajectory_shapes():
    Mlist, Glist, Slist = robot3()
    thetalist = torch.tensor([0.1, 0.1, 0.1])
    dthetalist = torch.tensor([0.1, 0.2, 0.3])
    taulist = torch.tensor([0.5, 0.6, 0.7])
    g = torch.tensor([0.0, 0.0, -9.8])
    Ftip = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    dd = pmr.ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist).squeeze(0)
    expected_dd = np_mr.ForwardDynamics(
        thetalist.cpu().numpy(), dthetalist.cpu().numpy(), taulist.cpu().numpy(),
        g.cpu().numpy(), Ftip.cpu().numpy(), Mlist.cpu().numpy(), Glist.cpu().numpy(), Slist.cpu().numpy()
    )
    assert np.allclose(dd.cpu().numpy(), expected_dd, atol=1e-2)

    thetamat = torch.stack([thetalist, thetalist + 0.01], dim=0)
    dthetamat = torch.zeros_like(thetamat)
    ddthetamat = torch.zeros_like(thetamat)
    Ftipmat = torch.ones((2, 6))
    taumat = pmr.InverseDynamicsTrajectory(thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist)
    assert taumat.shape == (1, 2, 3)

    th, dth = pmr.ForwardDynamicsTrajectory(thetalist, dthetalist, torch.ones((2, 3)), g, torch.ones((2, 6)), Mlist, Glist, Slist, 0.1, 2)
    assert th.shape == (1, 2, 3)
    assert dth.shape == (1, 2, 3)
