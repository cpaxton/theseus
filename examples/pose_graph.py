# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import theseus as th

import theseus.utils.examples as theg

torch.manual_seed(1)

file_path = "datasets/tinyGrid3D.g2o"
dtype = torch.float64

th.SO3.SO3_EPS = 1e-6

num_verts, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path, dtype=dtype)

objective = th.Objective(dtype)

for edge in edges:
    cost_func = th.Between(
        verts[edge.i], verts[edge.j], edge.weight, edge.relative_pose
    )
    objective.add(cost_func)

pose_prior = th.Difference(
    var=verts[0],
    cost_weight=th.ScaleCostWeight(torch.tensor(1e-6, dtype=dtype)),
    target=verts[0].copy(new_name=verts[0].name + "PRIOR"),
)
objective.add(pose_prior)

optimizer = th.LevenbergMarquardt(  # GaussNewton(
    objective,
    max_iterations=20,
    step_size=0.5,
)

theseus_optim = th.TheseusLayer(optimizer)

inputs = {var.name: var.data for var in verts}
theseus_optim.forward(inputs, optimizer_kwargs={"verbose": True})
