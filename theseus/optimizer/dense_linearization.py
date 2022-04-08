# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from theseus.core import Objective

from .linearization import Linearization
from .variable_ordering import VariableOrdering


class DenseLinearization(Linearization):
    def __init__(
        self,
        objective: Objective,
        ordering: Optional[VariableOrdering] = None,
        **kwargs
    ):
        super().__init__(objective, ordering)
        self.A: torch.Tensor = None
        self.AtA: torch.Tensor = None
        self.b: torch.Tensor = None
        self.Atb: torch.Tensor = None

    def _linearize_jacobian_impl(self):
        err_row_idx = 0
        self.A = torch.zeros(
            (self.objective.batch_size, self.num_rows, self.num_cols),
            device=self.objective.device,
            dtype=self.objective.dtype,
        )
        self.b = torch.zeros(
            (self.objective.batch_size, self.num_rows),
            device=self.objective.device,
            dtype=self.objective.dtype,
        )

        batch_size = self.objective.batch_size
        for (
            batch_cost_function,
            cost_functions,
        ) in self.objective.grouped_cost_functions.values():
            (
                batch_jacobians,
                batch_errors,
            ) = batch_cost_function.weighted_jacobians_error()
            # TODO: Implement FuncTorch
            batch_pos = 0
            for cost_function in cost_functions:
                # get the cost function jacobian/error slices from the batch
                jacobians = [
                    jacobian[batch_pos : batch_pos + batch_size]
                    for jacobian in batch_jacobians
                ]
                error = batch_errors[batch_pos : batch_pos + batch_size]
                batch_pos += batch_size

                # assign jacobians/error values to the correct place in A and b
                num_rows = cost_function.dim()
                row_slice = slice(err_row_idx, err_row_idx + num_rows)
                for var_idx_in_cost_function, var_jacobian in enumerate(jacobians):
                    var_idx_in_order = self.ordering.index_of(
                        cost_function.optim_var_at(var_idx_in_cost_function).name
                    )
                    var_start_col = self.var_start_cols[var_idx_in_order]

                    num_cols = var_jacobian.shape[2]
                    col_slice = slice(var_start_col, var_start_col + num_cols)
                    self.A[:, row_slice, col_slice] = var_jacobian

                self.b[:, row_slice] = -error
                err_row_idx += num_rows

    def _linearize_hessian_impl(self):
        self._linearize_jacobian_impl()
        At = self.A.transpose(1, 2)
        self.AtA = At.bmm(self.A)
        self.Atb = At.bmm(self.b.unsqueeze(2))

    def hessian_approx(self):
        return self.AtA
