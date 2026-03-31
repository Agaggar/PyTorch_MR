import numpy as np
import torch

import modern_robotics.core as np_mr
import modern_robotics.pytorch_mr.core as pmr


def test_time_scaling_examples():
    assert abs(float(pmr.CubicTimeScaling(2, 0.6)) - float(np_mr.CubicTimeScaling(2, 0.6))) < 1e-12
    assert abs(float(pmr.QuinticTimeScaling(2, 0.6)) - float(np_mr.QuinticTimeScaling(2, 0.6))) < 1e-12


def test_joint_trajectory_example_endpoints():
    start = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.0, 1.0])
    end = torch.tensor([1.2, 0.5, 0.6, 1.1, 2.0, 2.0, 0.9, 1.0])
    traj = pmr.JointTrajectory(start, end, 4, 6, 3).squeeze(0)
    assert torch.allclose(traj[0], start, atol=1e-6)
    assert torch.allclose(traj[-1], end, atol=1e-6)
    traj_np = np_mr.JointTrajectory(start.cpu().numpy(), end.cpu().numpy(), 4, 6, 3)
    assert np.allclose(traj.cpu().numpy(), traj_np, atol=1e-6)


def test_screw_and_cartesian_trajectory_shapes():
    Xstart = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0],
                           [0.0, 0.0, 0.0, 1.0]])
    Xend = torch.tensor([[0.0, 0.0, 1.0, 0.1],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 4.1],
                         [0.0, 0.0, 0.0, 1.0]])
    screw = pmr.ScrewTrajectory(Xstart, Xend, 5, 4, 3)
    cart = pmr.CartesianTrajectory(Xstart, Xend, 5, 4, 5)
    assert screw.shape == (1, 4, 4, 4)
    assert cart.shape == (1, 4, 4, 4)
    assert torch.allclose(screw[:, 0], Xstart.unsqueeze(0), atol=1e-6)
    assert torch.allclose(screw[:, -1], Xend.unsqueeze(0), atol=1e-5)
    screw_np = np_mr.ScrewTrajectory(Xstart.cpu().numpy(), Xend.cpu().numpy(), 5, 4, 3)
    cart_np = np_mr.CartesianTrajectory(Xstart.cpu().numpy(), Xend.cpu().numpy(), 5, 4, 5)
    assert np.allclose(screw.squeeze(0).cpu().numpy(), np.stack(screw_np, axis=0), atol=1e-5)
    assert np.allclose(cart.squeeze(0).cpu().numpy(), np.stack(cart_np, axis=0), atol=1e-5)


def test_computed_torque_and_simulate_control_smoke():
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
    thetalist = torch.tensor([0.1, 0.1, 0.1])
    dthetalist = torch.tensor([0.1, 0.2, 0.3])
    eint = torch.tensor([0.2, 0.2, 0.2])
    g = torch.tensor([0.0, 0.0, -9.8])
    tau = pmr.ComputedTorque(
        thetalist=thetalist, dthetalist=dthetalist, eint=eint, g=g, Mlist=Mlist, Glist=Glist, Slist=Slist,
        thetalistd=torch.tensor([1.0, 1.0, 1.0]), dthetalistd=torch.tensor([2.0, 1.2, 2.0]),
        ddthetalistd=torch.tensor([0.1, 0.1, 0.1]), Kp=1.3, Ki=1.2, Kd=1.1
    ).squeeze(0)
    expected_tau = np_mr.ComputedTorque(
        thetalist.cpu().numpy(), dthetalist.cpu().numpy(), eint.cpu().numpy(), g.cpu().numpy(),
        Mlist.cpu().numpy(), Glist.cpu().numpy(), Slist.cpu().numpy(),
        np.array([1.0, 1.0, 1.0]), np.array([2.0, 1.2, 2.0]), np.array([0.1, 0.1, 0.1]),
        1.3, 1.2, 1.1
    )
    assert np.allclose(tau.cpu().numpy(), expected_tau, atol=1e-2)

    traj = pmr.JointTrajectory(thetalist, torch.tensor([1.57, 3.14, 4.71]), 0.1, 5, 5).squeeze(0)
    dtraj = torch.zeros_like(traj)
    ddtraj = torch.zeros_like(traj)
    taumat, thetamat = pmr.SimulateControl(
        thetalist=thetalist, dthetalist=dthetalist, g=g, Ftipmat=torch.ones((5, 6)),
        Mlist=Mlist, Glist=Glist, Slist=Slist, thetamatd=traj, dthetamatd=dtraj, ddthetamatd=ddtraj,
        gtilde=g, Mtildelist=Mlist, Gtildelist=Glist, Kp=20, Ki=10, Kd=18, dt=0.02, intRes=2
    )
    assert taumat.shape == (1, 5, 3)
    assert thetamat.shape == (1, 5, 3)
