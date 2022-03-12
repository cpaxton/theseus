from typing import List, Optional, Tuple

import torch

import theseus as th

from .util import soft_loss_huber_like


class ReprojectionError(th.CostFunction):
    def __init__(
        self,
        camera_pose: th.SE3,
        loss_radius: th.Vector,
        world_point: th.Point3,
        image_feature_point: th.Point2,
        focal_length: th.Vector,
        calib_k1: th.Vector = None,
        calib_k2: th.Vector = None,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=camera_pose.dtype))
        super().__init__(
            cost_weight=weight,
            name=name,
        )
        self.camera_pose = camera_pose
        self.loss_radius = loss_radius
        self.focal_length = focal_length
        self.calib_k1 = calib_k1
        self.calib_k2 = calib_k2
        batch_size = self.camera_pose.shape[0]
        if self.calib_k1 is None:
            self.calib_k1 = th.Vector(
                data=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name="calib_k1",
            )
        if self.calib_k2 is None:
            self.calib_k2 = th.Vector(
                data=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name="calib_k2",
            )
        self.world_point = world_point
        self.image_feature_point = image_feature_point

        self.register_optim_vars(["camera_pose", "world_point"])
        self.register_aux_vars(["loss_radius", "focal_length", "image_feature_point"])

    def error(self) -> torch.Tensor:
        point_cam = self.camera_pose.transform_from(self.world_point)
        proj = point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.data * (
            1.0 + proj_sqn * (self.calib_k1.data + proj_sqn * self.calib_k2.data)
        )
        point_projection = proj * proj_factor

        err = point_projection - self.image_feature_point.data

        err_norm = torch.norm(err, dim=1).unsqueeze(1)
        exp_loss = torch.exp(self.loss_radius.data)

        val, _ = soft_loss_huber_like(err_norm, exp_loss)
        return val

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cpose_wpt_jacs = []
        point_cam = self.camera_pose.transform_from(self.world_point, cpose_wpt_jacs)
        J = torch.cat(cpose_wpt_jacs, dim=2)

        proj = point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.data * (
            1.0 + proj_sqn * (self.calib_k1.data + proj_sqn * self.calib_k2.data)
        )
        d_proj_factor = self.focal_length.data * (
            self.calib_k1.data + 2.0 * proj_sqn * self.calib_k2.data
        )
        point_projection = proj * proj_factor

        # derivative of N/D is (N' - ND'/D) / D
        d_num = J[:, 0:2, :]
        num_dden_den = torch.bmm(
            point_cam[:, :2].unsqueeze(2),
            (J[:, 2, :] / point_cam[:, 2:3]).unsqueeze(1),
        )
        proj_jac = (d_num - num_dden_den) / point_cam[:, 2:].unsqueeze(2)
        proj_sqn_jac = 2.0 * proj.unsqueeze(2) * torch.bmm(proj.unsqueeze(1), proj_jac)
        point_projection_jac = proj_jac * proj_factor.unsqueeze(
            2
        ) + proj_sqn_jac * d_proj_factor.unsqueeze(2)

        err = point_projection - self.image_feature_point.data
        err_norm = torch.norm(err, dim=1).unsqueeze(1)
        err_dir = err / err_norm
        norm_jac = torch.bmm(err_dir.unsqueeze(1), point_projection_jac)
        exp_loss = torch.exp(self.loss_radius.data)

        val, der = soft_loss_huber_like(err_norm, exp_loss)
        soft_jac = norm_jac * der.unsqueeze(1)

        return [soft_jac[:, :, :6], soft_jac[:, :, 6:]], val

    def dim(self) -> int:
        return 2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self):
        return ReprojectionError(
            self.camera_pose,
            self.loss_radius,
            self.focal_length,
            self.world_point,
            self.image_feature_point,
            weight=self.weight,
            name=self.name,
        )
