# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th


class ResidualCostFunction(th.CostFunction):
    def __init__(
        self,
        optim_vars,
        cost_weight,
        name=None,
        true_coeffs=None,
        point=None,
        multivar=False,
        noise_mag=0,
    ):
        super().__init__(cost_weight, name=name)
        len_vars = len(optim_vars) if multivar else optim_vars[0].dof()
        assert true_coeffs.ndim == 1 and true_coeffs.numel() == len_vars
        if point.ndim == 1:
            point = point.unsqueeze(0)
        assert point.ndim == 2 and point.shape[1] == len_vars - 1
        batch_size = point.shape[0]
        for i, var in enumerate(optim_vars):
            attr_name = f"optim_var_{i}"
            setattr(self, attr_name, var)
            self.register_optim_var(attr_name)

        self.point = torch.cat([point, torch.ones(batch_size, 1)], dim=1)
        self.target = (self.point * true_coeffs.unsqueeze(0)).sum(1, keepdim=True) ** 2
        if noise_mag:
            self.target += noise_mag * torch.randn(size=self.target.shape)
        self.multivar = multivar

    def _eval_coeffs(self):
        if self.multivar:
            coeffs = torch.cat([v.data for v in self.optim_vars], axis=1)
        else:
            coeffs = self.optim_var_0.data
        return (self.point * coeffs).sum(1, keepdim=True)

    def error(self):
        # h(B * x) - h(Btrue * x)
        return self._eval_coeffs() ** 2 - self.target

    def jacobians(self):
        dhdz = 2 * self._eval_coeffs()
        grad = self.point * dhdz
        return [grad.unsqueeze(1)], self.error()

    def dim(self):
        return 1

    def _copy_impl(self, new_name=None):
        raise NotImplementedError


def _check_info(info, batch_size, max_iterations, initial_error, objective):
    assert info.err_history.shape == (batch_size, max_iterations + 1)
    assert info.err_history[:, 0].allclose(initial_error)
    assert info.err_history.argmin(dim=1).allclose(info.best_iter + 1)
    last_error = objective.error_squared_norm() / 2
    last_convergence_idx = info.converged_iter.max().item()
    assert info.err_history[:, last_convergence_idx].allclose(last_error)


# This test uses least-squares regression to find the coefficients b_j of
#
#       y = f(x, B) = h(sum_i b_i * x_i + b_0)  ,  for i in [1, nvars-1]
#
# where h(z) = z^2.
#
# There is only one nvars-dimensional Vector for all coefficients, and
# one cost function per data-point (generated at random), which computes residual
#
#       (f(p, Btrue) - f(p, B))
#
# for some data point p specific to this cost function. There is no noise.
def _check_nonlinear_least_squares_fit(
    nonlinear_optim_cls,
    optimize_kwargs,
    points,
    nvars,
    npoints,
    batch_size,
):
    true_coeffs = torch.ones(nvars)
    variables = [th.Vector(nvars, name="coefficients")]
    cost_weight = th.ScaleCostWeight(1.0)
    objective = th.Objective()
    for i in range(npoints):
        objective.add(
            ResidualCostFunction(
                variables,
                cost_weight,
                true_coeffs=true_coeffs,
                point=points[i],
                name=f"residual_point_{i}",
            )
        )

    # Initial value is B = [0, 1, ..., nvars - 1]
    values = {"coefficients": torch.arange(nvars).repeat(batch_size, 1).float()}
    objective.update(values)
    initial_error = objective.error_squared_norm() / 2
    max_iterations = 20
    optimizer = nonlinear_optim_cls(objective)
    assert isinstance(optimizer.linear_solver, th.CholeskyDenseSolver)
    optimizer.set_params(max_iterations=max_iterations)
    info = optimizer.optimize(
        track_best_solution=True, track_err_history=True, **optimize_kwargs
    )
    # Solution must now match the true coefficients
    assert variables[0].data.allclose(true_coeffs.repeat(batch_size, 1), atol=1e-6)
    _check_info(info, batch_size, max_iterations, initial_error, objective)


# Same test as above but now each cost function has m 1-d variables instead
# of a single m-dim variable
def _check_nonlinear_least_squares_fit_multivar(
    nonlinear_optim_cls,
    optimize_kwargs,
    points,
    nvars,
    npoints,
    batch_size,
):
    true_coeffs = torch.ones(nvars)
    variables = [th.Vector(1, name=f"coeff{i}") for i in range(nvars)]
    cost_weight = th.ScaleCostWeight(1.0)
    objective = th.Objective()
    for i in range(npoints):
        objective.add(
            ResidualCostFunction(
                variables,
                cost_weight,
                true_coeffs=true_coeffs,
                point=points[i],
                name=f"residual_point_{i}",
                multivar=True,
            )
        )
    # Initial value is B = [0, 1, ..., nvars - 1]
    values = dict((f"coeff{i}", i * torch.ones(batch_size, 1)) for i in range(nvars))
    objective.update(values)
    initial_error = objective.error_squared_norm() / 2

    max_iterations = 20
    optimizer = nonlinear_optim_cls(objective)
    assert isinstance(optimizer.linear_solver, th.CholeskyDenseSolver)
    optimizer.set_params(max_iterations=max_iterations)
    info = optimizer.optimize(
        track_best_solution=True, track_err_history=True, **optimize_kwargs
    )

    # Solution must now match the true coefficients
    for i in range(nvars):
        assert variables[i].data.allclose(true_coeffs[i].repeat(batch_size, 1))

    _check_info(info, batch_size, max_iterations, initial_error, objective)


def _check_optimizer_returns_fail_status_on_singular(
    nonlinear_optim_cls, optimize_kwargs
):
    class BadLinearization(th.optimizer.Linearization):
        def __init__(self, objective):
            super().__init__(objective)

        def _linearize_jacobian_impl(self):
            pass

        def _linearize_hessian_impl(self):
            self.AtA = torch.zeros(
                self.objective.batch_size, self.num_rows, self.num_cols
            )
            self.Atb = torch.ones(self.objective.batch_size, self.num_cols)

    class MockCostFunction(th.CostFunction):
        def __init__(self, optim_vars, cost_weight):
            super().__init__(cost_weight)
            for var in optim_vars:
                setattr(self, var.name, var)
                self.register_optim_var(var.name)

        def error(self):
            return torch.ones(1)

        def jacobians(self):
            return torch.ones(1), self.error()

        def dim(self):
            return 1

        def _copy_impl(self):
            raise NotImplementedError

    objective = th.Objective()
    variables = [th.Vector(1, name="dummy")]
    objective.add(MockCostFunction(variables, th.ScaleCostWeight(1.0)))
    values = {"dummy": torch.zeros(1, 1)}
    objective.update(values)

    optimizer = nonlinear_optim_cls(objective)
    assert isinstance(optimizer.linear_solver, th.CholeskyDenseSolver)
    optimizer.set_params(max_iterations=30)
    optimizer.linear_solver.linearization = BadLinearization(objective)
    with pytest.raises(RuntimeError):
        optimizer.optimize(track_best_solution=True, verbose=True, **optimize_kwargs)

    with pytest.warns(RuntimeWarning):
        with torch.no_grad():
            info = optimizer.optimize(
                track_best_solution=True, track_err_history=True, **optimize_kwargs
            )
        assert (info.status == th.NonlinearOptimizerStatus.FAIL).all()


def run_nonlinear_least_squares_check(
    nonlinear_optim_cls, optimize_kwargs, singular_check=True
):
    rng = torch.Generator()
    rng.manual_seed(0)
    nvars = 5
    npoints = 50
    batch_size = 32
    points = [torch.randn(batch_size, nvars - 1, generator=rng) for _ in range(npoints)]

    print("----- Check single variable formulation -----")
    _check_nonlinear_least_squares_fit(
        nonlinear_optim_cls,
        optimize_kwargs,
        points,
        nvars,
        npoints,
        batch_size,
    )
    print("----- Check multiple variable formulation -----")
    _check_nonlinear_least_squares_fit_multivar(
        nonlinear_optim_cls,
        optimize_kwargs,
        points,
        nvars,
        npoints,
        batch_size,
    )
    if singular_check:
        print("----- Check failure status on singular system formulation -----")
        _check_optimizer_returns_fail_status_on_singular(
            nonlinear_optim_cls, optimize_kwargs
        )
