import numpy as np
import torch

import modern_robotics.core as np_mr
import modern_robotics.pytorch_mr.core as pmr


def test_fkin_body_example():
    M = torch.tensor([[-1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 6.0],
                      [0.0, 0.0, -1.0, 2.0],
                      [0.0, 0.0, 0.0, 1.0]])
    Blist = torch.tensor([[0.0, 0.0, -1.0, 2.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.1]]).T
    thetalist = torch.tensor([torch.pi / 2.0, 3.0, torch.pi])
    out = pmr.FKinBody(M, Blist, thetalist).squeeze(0).cpu().numpy()
    expected = np_mr.FKinBody(M.cpu().numpy(), Blist.cpu().numpy(), thetalist.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-5)


def test_fkin_space_example():
    M = torch.tensor([[-1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 6.0],
                      [0.0, 0.0, -1.0, 2.0],
                      [0.0, 0.0, 0.0, 1.0]])
    Slist = torch.tensor([[0.0, 0.0, 1.0, 4.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, -1.0, -6.0, 0.0, -0.1]]).T
    thetalist = torch.tensor([torch.pi / 2.0, 3.0, torch.pi])
    out = pmr.FKinSpace(M, Slist, thetalist).squeeze(0).cpu().numpy()
    expected = np_mr.FKinSpace(M.cpu().numpy(), Slist.cpu().numpy(), thetalist.cpu().numpy())
    assert np.allclose(out, expected, atol=1e-5)


def test_jacobians_shapes():
    Blist = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.2, 0.2],
                          [1.0, 0.0, 0.0, 2.0, 0.0, 3.0],
                          [0.0, 1.0, 0.0, 0.0, 2.0, 1.0],
                          [1.0, 0.0, 0.0, 0.2, 0.3, 0.4]]).T
    thetalist = torch.tensor([0.2, 1.1, 0.1, 1.2])
    Jb = pmr.JacobianBody(Blist, thetalist)
    Js = pmr.JacobianSpace(Blist, thetalist)
    assert Jb.shape == (1, 6, 4)
    assert Js.shape == (1, 6, 4)

    Jb_np = np_mr.JacobianBody(Blist.cpu().numpy(), thetalist.cpu().numpy())
    Js_np = np_mr.JacobianSpace(Blist.cpu().numpy(), thetalist.cpu().numpy())
    assert np.allclose(Jb.squeeze(0).cpu().numpy(), Jb_np, atol=1e-5)
    assert np.allclose(Js.squeeze(0).cpu().numpy(), Js_np, atol=1e-5)


def test_ikin_body_and_space_converge():
    Blist = torch.tensor([[0.0, 0.0, -1.0, 2.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.1]]).T
    Slist = torch.tensor([[0.0, 0.0, 1.0, 4.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, -1.0, -6.0, 0.0, -0.1]]).T
    M = torch.tensor([[-1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 6.0],
                      [0.0, 0.0, -1.0, 2.0],
                      [0.0, 0.0, 0.0, 1.0]])
    T = torch.tensor([[0.0, 1.0, 0.0, -5.0],
                      [1.0, 0.0, 0.0, 4.0],
                      [0.0, 0.0, -1.0, 1.6858],
                      [0.0, 0.0, 0.0, 1.0]])
    guess = torch.tensor([1.5, 2.5, 3.0])
    th_b, ok_b = pmr.IKinBody(Blist, M, T, guess, eomg=0.01, ev=0.001)
    th_s, ok_s = pmr.IKinSpace(Slist, M, T, guess, eomg=0.01, ev=0.001)
    assert bool(ok_b.squeeze(0))
    assert bool(ok_s.squeeze(0))

    th_b_np, ok_b_np = np_mr.IKinBody(Blist.cpu().numpy(), M.cpu().numpy(), T.cpu().numpy(), guess.cpu().numpy(), 0.01, 0.001)
    th_s_np, ok_s_np = np_mr.IKinSpace(Slist.cpu().numpy(), M.cpu().numpy(), T.cpu().numpy(), guess.cpu().numpy(), 0.01, 0.001)
    assert bool(ok_b_np)
    assert bool(ok_s_np)
    assert np.allclose(th_b.squeeze(0).cpu().numpy(), th_b_np, atol=1e-3)
    assert np.allclose(th_s.squeeze(0).cpu().numpy(), th_s_np, atol=1e-3)
