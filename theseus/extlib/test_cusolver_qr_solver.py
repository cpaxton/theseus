import numpy as np
import pytest  # noqa: F401
import torch  # needed for import of Torch C++ extensions to work
from scipy.sparse import csr_matrix


def test_lu_solver(verbose=False):
    if not torch.cuda.is_available():
        return
    from theseus.extlib.cusolver_qr_solver import CusolverQRSolver

    # 4x3 mat pattern:
    #  1 1
    #    1 1
    #    1
    #  1   1

    batch_size = 3
    A_rows = 4
    A_cols = 3
    A_nnz = 7
    A_rowPtr = torch.tensor([0, 2, 4, 5, 7], dtype=torch.int).cuda()
    A_colInd = torch.tensor([0, 1, 1, 2, 1, 0, 2], dtype=torch.int).cuda()
    A_val = torch.rand((batch_size, A_nnz), dtype=torch.double).cuda()
    b = torch.rand((batch_size, A_rows), dtype=torch.double).cuda()

    A_csr = [
        csr_matrix((A_val[i].cpu(), A_colInd.cpu(), A_rowPtr.cpu()), (A_rows, A_cols))
        for i in range(batch_size)
    ]

    solver = CusolverQRSolver(A_cols, A_rowPtr, A_colInd)
    x = solver.factor_and_solve(batch_size, A_val, b)

    residuals = [
        A_csr[i].T @ (A_csr[i] @ x[i].cpu().numpy() - b[i].cpu().numpy())
        for i in range(batch_size)
    ]

    assert all(np.linalg.norm(res) < 1e-10 for res in residuals)
