# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import torch

import theseus
import theseus.constants

from .lie_group import LieGroup
from .point_types import Point3
from .so3 import SO3


class SE3(LieGroup):
    def __init__(
        self,
        x_y_z_quaternion: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        requires_check: bool = True,
    ):
        if x_y_z_quaternion is not None and data is not None:
            raise ValueError("Please provide only one of x_y_z_quaternion or data.")
        if x_y_z_quaternion is not None:
            dtype = x_y_z_quaternion.dtype
        if data is not None and requires_check:
            self._SE3_matrix_check(data)
        super().__init__(data=data, name=name, dtype=dtype)
        if x_y_z_quaternion is not None:
            self.update_from_x_y_z_quaternion(x_y_z_quaternion=x_y_z_quaternion)

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SE3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        ret = SE3()
        rotation = SO3.rand(
            size[0],
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        translation = Point3.rand(
            size[0],
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        ret.update_from_rot_and_trans(rotation=rotation, translation=translation)
        return ret

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SE3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        ret = SE3()
        rotation = SO3.randn(
            size[0],
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        translation = Point3.randn(
            size[0],
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        ret.update_from_rot_and_trans(rotation=rotation, translation=translation)
        return ret

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        return torch.eye(3, 4).view(1, 3, 4)

    def dof(self) -> int:
        return 6

    def __repr__(self) -> str:
        return f"SE3(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            return f"SE3(matrix={self.data}), name={self.name})"

    def _adjoint_impl(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 6, 6).to(dtype=self.dtype, device=self.device)
        ret[:, :3, :3] = self[:, :3, :3]
        ret[:, 3:, 3:] = self[:, :3, :3]
        ret[:, :3, 3:] = SO3.hat(self[:, :3, 3]) @ self[:, :3, :3]

        return ret

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        self._project_check(euclidean_grad, is_sparse)
        ret = torch.zeros(
            euclidean_grad.shape[:-2] + torch.Size([6]),
            dtype=self.dtype,
            device=self.device,
        )

        if is_sparse:
            temp = torch.einsum(
                "i...jk,i...jl->i...lk", euclidean_grad, self.data[:, :, :3]
            )
        else:
            temp = torch.einsum(
                "...jk,...ji->...ik", euclidean_grad, self.data[:, :, :3]
            )

        ret[..., :3] = temp[..., 3]
        ret[..., 3] = temp[..., 2, 1] - temp[..., 1, 2]
        ret[..., 4] = temp[..., 0, 2] - temp[..., 2, 0]
        ret[..., 5] = temp[..., 1, 0] - temp[..., 0, 1]

        return ret

    @staticmethod
    def _SE3_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 4):
            raise ValueError("SE(3) can only be 3x4 matrices.")
        SO3._SO3_matrix_check(matrix.data[:, :3, :3])

    @staticmethod
    def x_y_z_unit_quaternion_to_SE3(x_y_z_quaternion: torch.Tensor) -> "SE3":
        if x_y_z_quaternion.ndim == 1:
            x_y_z_quaternion = x_y_z_quaternion.unsqueeze(0)

        if x_y_z_quaternion.ndim != 2 and x_y_z_quaternion.shape[1] != 7:
            raise ValueError("x_y_z_quaternion can only be 7-D vectors.")

        ret = SE3()

        batch_size = x_y_z_quaternion.shape[0]
        ret.data = torch.empty(batch_size, 3, 4).to(
            device=x_y_z_quaternion.device, dtype=x_y_z_quaternion.dtype
        )
        ret[:, :, :3] = SO3.unit_quaternion_to_SO3(x_y_z_quaternion[:, 3:]).data
        ret[:, :, 3] = x_y_z_quaternion[:, :3]

        return ret

    @staticmethod
    def _hat_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
            raise ValueError("Hat matrices of SE(3) can only be 4x4 matrices")

        if matrix[:, 3].abs().max().item() > theseus.constants.EPS:
            raise ValueError("The last row of hat matrices of SE(3) can only be zero.")

        if (
            matrix[:, :3, :3].transpose(1, 2) + matrix[:, :3, :3]
        ).abs().max().item() > theseus.constants.EPS:
            raise ValueError(
                "The 3x3 top-left corner of hat matrices of SE(3) can only be skew-symmetric."
            )

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "SE3":
        if tangent_vector.ndim != 2 or tangent_vector.shape[1] != 6:
            raise ValueError("Tangent vectors of SE(3) can only be 6-D vectors.")

        NEAR_ZERO_EPS = 5e-3

        ret = SE3(dtype=tangent_vector.dtype)

        tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
        tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

        theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
        theta2 = theta**2
        theta3 = theta**3

        near_zero = theta < NEAR_ZERO_EPS
        non_zero = torch.ones(
            1, dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        theta_nz = torch.where(near_zero, non_zero, theta)
        theta2_nz = torch.where(near_zero, non_zero, theta2)
        theta3_nz = torch.where(near_zero, non_zero, theta3)

        # Compute the rotation
        sine = theta.sin()
        cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
        sine_by_theta = torch.where(
            near_zero, 0.5 * cosine + 0.5, theta.sin() / theta_nz
        )
        one_minus_cosine_by_theta2 = torch.where(
            near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
        )
        ret.data = torch.zeros(tangent_vector.shape[0], 3, 4).to(
            dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        ret.data[:, :3, :3] = (
            one_minus_cosine_by_theta2
            * tangent_vector_ang
            @ tangent_vector_ang.transpose(1, 2)
        )

        ret[:, 0, 0] += cosine.view(-1)
        ret[:, 1, 1] += cosine.view(-1)
        ret[:, 2, 2] += cosine.view(-1)
        temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
        ret[:, 0, 1] -= temp[:, 2]
        ret[:, 1, 0] += temp[:, 2]
        ret[:, 0, 2] += temp[:, 1]
        ret[:, 2, 0] -= temp[:, 1]
        ret[:, 1, 2] -= temp[:, 0]
        ret[:, 2, 1] += temp[:, 0]

        # Compute the translation
        sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
        one_minus_cosine_by_theta2 = torch.where(
            near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2
        )
        theta_minus_sine_by_theta3_t = torch.where(
            near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz
        )

        ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
        ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(
            tangent_vector_ang, tangent_vector_lin
        )
        ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
            tangent_vector_ang
            @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
        )

        if jacobians is not None:
            SE3._check_jacobians_list(jacobians)
            theta3_nz = theta_nz * theta2_nz
            theta_minus_sine_by_theta3_rot = torch.where(
                near_zero, torch.zeros_like(theta), theta_minus_sine_by_theta3_t
            )
            jac = torch.zeros(
                tangent_vector.shape[0],
                6,
                6,
                dtype=tangent_vector.dtype,
                device=tangent_vector.device,
            )
            jac[:, :3, :3] = (
                theta_minus_sine_by_theta3_rot
                * tangent_vector_ang.view(-1, 3, 1)
                @ tangent_vector_ang.view(-1, 1, 3)
            )
            diag_jac = jac.diagonal(dim1=1, dim2=2)
            diag_jac += sine_by_theta.view(-1, 1)

            jac_temp_rot = one_minus_cosine_by_theta2.view(
                -1, 1
            ) * tangent_vector_ang.view(-1, 3)

            jac[:, 0, 1] += jac_temp_rot[:, 2]
            jac[:, 1, 0] -= jac_temp_rot[:, 2]
            jac[:, 0, 2] -= jac_temp_rot[:, 1]
            jac[:, 2, 0] += jac_temp_rot[:, 1]
            jac[:, 1, 2] += jac_temp_rot[:, 0]
            jac[:, 2, 1] -= jac_temp_rot[:, 0]

            jac[:, 3:, 3:] = jac[:, :3, :3]

            minus_one_by_twelve = torch.tensor(
                -1 / 12.0,
                dtype=sine_by_theta.dtype,
                device=sine_by_theta.device,
            )
            d_one_minus_cosine_by_theta2 = torch.where(
                near_zero,
                minus_one_by_twelve,
                (sine_by_theta - 2 * one_minus_cosine_by_theta2) / theta2_nz,
            )
            minus_one_by_sixty = torch.tensor(
                -1 / 60.0,
                dtype=one_minus_cosine_by_theta2.dtype,
                device=one_minus_cosine_by_theta2.device,
            )
            d_theta_minus_sine_by_theta3 = torch.where(
                near_zero,
                minus_one_by_sixty,
                (one_minus_cosine_by_theta2 - 3 * theta_minus_sine_by_theta3_t)
                / theta2_nz,
            )

            w = tangent_vector[:, 3:]
            v = tangent_vector[:, :3]
            wv = w.cross(v)
            wwv = w.cross(wv)
            sw = theta_minus_sine_by_theta3_t.view(-1, 1) * w

            jac_temp_t = (
                d_one_minus_cosine_by_theta2.view(-1, 1) * wv
                + d_theta_minus_sine_by_theta3.view(-1, 1) * wwv
            ).view(-1, 3, 1) @ w.view(-1, 1, 3)
            jac_temp_t -= v.view(-1, 3, 1) @ sw.view(-1, 1, 3)
            jac_temp_v = (
                -one_minus_cosine_by_theta2.view(-1, 1) * v
                - theta_minus_sine_by_theta3_t.view(-1, 1) * wv
            )
            jac_temp_t += SO3.hat(jac_temp_v)
            diag_jac_t = torch.diagonal(jac_temp_t, dim1=1, dim2=2)
            diag_jac_t += (sw.view(-1, 1, 3) @ v.view(-1, 3, 1)).view(-1, 1)

            jac[:, :3, 3:] = ret[:, :, :3].transpose(1, 2) @ jac_temp_t

            jacobians.append(jac)

        return ret

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        NEAR_PI_EPS = 1e-7
        NEAR_ZERO_EPS = 5e-3

        sine_axis = torch.zeros(self.shape[0], 3, dtype=self.dtype, device=self.device)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        cosine = 0.5 * (self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2] - 1)
        sine = sine_axis.norm(dim=1)
        theta = torch.atan2(sine, cosine)
        theta2 = theta**2
        non_zero = torch.ones(1, dtype=self.dtype, device=self.device)

        near_zero = theta < NEAR_ZERO_EPS

        # Compute the rotation
        not_near_pi = 1 + cosine > NEAR_PI_EPS
        # theta is not near pi
        near_zero_not_near_pi = near_zero[not_near_pi]
        # Compute the approximation of theta / sin(theta) when theta is near to 0
        sine_nz = torch.where(near_zero_not_near_pi, non_zero, sine[not_near_pi])
        scale = torch.where(
            near_zero_not_near_pi,
            1 + sine[not_near_pi] ** 2 / 6,
            theta[not_near_pi] / sine_nz,
        )
        ret_ang = torch.zeros_like(sine_axis)
        ret_ang[not_near_pi] = sine_axis[not_near_pi] * scale.view(-1, 1)

        # theta is near pi
        near_pi = ~not_near_pi
        ddiag = torch.diagonal(self[near_pi], dim1=1, dim2=2)
        # Find the index of major coloumns and diagonals
        major = torch.logical_and(
            ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
        ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
        sel_rows = 0.5 * (self[near_pi, major, :3] + self[near_pi, :3, major])
        aux = torch.ones(sel_rows.shape[0], dtype=torch.bool)
        sel_rows[aux, major] -= cosine[near_pi]
        axis = sel_rows / sel_rows.norm(dim=1, keepdim=True)
        ret_ang[near_pi] = axis * (
            theta[near_pi] * sine_axis[near_pi, major].sign()
        ).view(-1, 1)

        # Compute the translation
        sine_theta = sine * theta
        two_cosine_minus_two = 2 * cosine - 2
        two_cosine_minus_two_nz = torch.where(near_zero, non_zero, two_cosine_minus_two)

        theta2_nz = torch.where(near_zero, non_zero, theta2)

        a = torch.where(
            near_zero, 1 - theta2 / 12, -sine_theta / two_cosine_minus_two_nz
        )
        b = torch.where(
            near_zero,
            1.0 / 12 + theta2 / 720,
            (sine_theta + two_cosine_minus_two) / (theta2_nz * two_cosine_minus_two_nz),
        )

        translation = self[:, :, 3].view(-1, 3, 1)
        ret_lin = a.view(-1, 1) * self[:, :, 3]
        ret_lin -= 0.5 * torch.cross(ret_ang, self[:, :, 3])
        ret_ang_ext = ret_ang.view(-1, 3, 1)
        ret_lin += b.view(-1, 1) * (
            ret_ang_ext @ (ret_ang_ext.transpose(1, 2) @ translation)
        ).view(-1, 3)

        if jacobians is not None:
            SE3._check_jacobians_list(jacobians)
            jac = torch.zeros(self.shape[0], 6, 6, dtype=self.dtype, device=self.device)

            b_ret_ang = b.view(-1, 1) * ret_ang
            jac[:, :3, :3] = b_ret_ang.view(-1, 3, 1) * ret_ang.view(-1, 1, 3)

            half_ret_ang = 0.5 * ret_ang
            jac[:, 0, 1] -= half_ret_ang[:, 2]
            jac[:, 1, 0] += half_ret_ang[:, 2]
            jac[:, 0, 2] += half_ret_ang[:, 1]
            jac[:, 2, 0] -= half_ret_ang[:, 1]
            jac[:, 1, 2] -= half_ret_ang[:, 0]
            jac[:, 2, 1] += half_ret_ang[:, 0]

            diag_jac_rot = torch.diagonal(jac[:, :3, :3], dim1=1, dim2=2)
            diag_jac_rot += a.view(-1, 1)

            jac[:, 3:, 3:] = jac[:, :3, :3]

            theta_nz = torch.where(near_zero, non_zero, theta)
            theta4_nz = theta2_nz**2
            c = torch.where(
                near_zero,
                -1 / 360.0 - theta2 / 7560.0,
                -(2 * two_cosine_minus_two + theta * sine + theta2)
                / (theta4_nz * two_cosine_minus_two_nz),
            )
            d = torch.where(
                near_zero,
                -1 / 6.0 - theta2 / 180.0,
                (theta - sine) / (theta_nz * two_cosine_minus_two_nz),
            )
            e = (ret_ang.view(-1, 1, 3) @ ret_lin.view(-1, 3, 1)).view(-1)

            ce_ret_ang = (c * e).view(-1, 1) * ret_ang
            jac[:, :3, 3:] = ce_ret_ang.view(-1, 3, 1) * ret_ang.view(-1, 1, 3)
            jac[:, :3, 3:] += b_ret_ang.view(-1, 3, 1) * ret_lin.view(
                -1, 1, 3
            ) + ret_lin.view(-1, 3, 1) * b_ret_ang.view(-1, 1, 3)
            diag_jac_t = torch.diagonal(jac[:, :3, 3:], dim1=1, dim2=2)
            diag_jac_t += (e * d).view(-1, 1)

            half_ret_lin = 0.5 * ret_lin
            jac[:, 0, 4] -= half_ret_lin[:, 2]
            jac[:, 1, 3] += half_ret_lin[:, 2]
            jac[:, 0, 5] += half_ret_lin[:, 1]
            jac[:, 2, 3] -= half_ret_lin[:, 1]
            jac[:, 1, 5] -= half_ret_lin[:, 0]
            jac[:, 2, 4] += half_ret_lin[:, 0]

            jacobians.append(jac)

        return torch.cat([ret_lin, ret_ang], dim=1)

    def _compose_impl(self, se3_2: LieGroup) -> "SE3":
        se3_2 = cast(SE3, se3_2)
        batch_size = max(self.shape[0], se3_2.shape[0])
        ret = SE3()
        ret.data = torch.zeros(batch_size, 3, 4, dtype=self.dtype, device=self.device)
        ret[:, :, :3] = self[:, :, :3] @ se3_2[:, :, :3]
        ret[:, :, 3] = self[:, :, 3]
        ret[:, :, 3:] += self[:, :, :3] @ se3_2[:, :, 3:]

        return ret

    def _inverse_impl(self, get_jacobian: bool = False) -> "SE3":
        ret = torch.zeros(self.shape[0], 3, 4).to(dtype=self.dtype, device=self.device)
        rotT = self.data[:, :3, :3].transpose(1, 2)
        ret[:, :, :3] = rotT
        ret[:, :, 3] = -(rotT @ self.data[:, :3, 3].unsqueeze(2)).view(-1, 3)
        # if self.data is a valid SE(3), so is the inverse
        return SE3(data=ret, requires_check=False)

    def to_matrix(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 4, 4).to(dtype=self.dtype, device=self.device)
        ret[:, :3] = self.data
        ret[:, 3, 3] = 1
        return ret

    def update_from_x_y_z_quaternion(self, x_y_z_quaternion: torch.Tensor):
        self.update(SE3.x_y_z_unit_quaternion_to_SE3(x_y_z_quaternion))

    def update_from_rot_and_trans(self, rotation: SO3, translation: Point3):
        if rotation.shape[0] != translation.shape[0]:
            raise ValueError("rotation and translation must have the same batch size.")

        if rotation.dtype != translation.dtype:
            raise ValueError("rotation and translation must be of the same type.")

        if rotation.device != translation.device:
            raise ValueError("rotation and translation must be on the same device.")

        self.data = torch.cat((rotation.data, translation.data.unsqueeze(2)), dim=2)

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        _check = tangent_vector.ndim == 2 and tangent_vector.shape[1] == 6
        if not _check:
            raise ValueError("Invalid vee matrix for SE(3).")
        matrix = torch.zeros(tangent_vector.shape[0], 4, 4).to(
            dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        matrix[:, :3, :3] = SO3.hat(tangent_vector[:, 3:])
        matrix[:, :3, 3] = tangent_vector[:, :3]

        return matrix

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        SE3._hat_matrix_check(matrix)
        return torch.cat((matrix[:, :3, 3], SO3.vee(matrix[:, :3, :3])), dim=1)

    def _copy_impl(self, new_name: Optional[str] = None) -> "SE3":
        # if self.data is a valid SE(3), so is the copy
        return SE3(data=self.data.clone(), name=new_name, requires_check=False)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SE3":
        return cast(SE3, super().copy(new_name=new_name))

    def _transform_shape_check(self, point: Union[Point3, torch.Tensor]):
        err_msg = (
            f"SE3 can only transform vectors of shape [{self.shape[0]}, 3] or [1, 3], "
            f"but the input has shape {point.shape}."
        )

        if isinstance(point, torch.Tensor):
            if not point.ndim == 2 or point.shape[1] != 3:
                raise ValueError(err_msg)
        elif point.dof() != 3:
            raise ValueError(err_msg)
        if (
            point.shape[0] != self.shape[0]
            and point.shape[0] != 1
            and self.shape[0] != 1
        ):
            raise ValueError(err_msg)

    def transform_from(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._transform_shape_check(point)
        batch_size = max(self.shape[0], point.shape[0])
        if isinstance(point, torch.Tensor):
            p = point.view(-1, 3, 1)
        else:
            p = point.data.view(-1, 3, 1)

        ret = Point3(data=(self[:, :, :3] @ p).view(-1, 3))
        ret.data += self[:, :, 3]

        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            # Right jacobians for SE(3) are computed
            Jg = torch.zeros(batch_size, 3, 6, dtype=self.dtype, device=self.device)
            Jg[:, :, :3] = self[:, :, :3]
            Jg[:, :, 3:] = -self[:, :, :3] @ SO3.hat(p)
            # Jacobians for point
            Jpnt = Jg[:, :, :3]

            jacobians.extend([Jg, Jpnt])

        return ret

    def transform_to(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._transform_shape_check(point)
        batch_size = max(self.shape[0], point.shape[0])
        if isinstance(point, torch.Tensor):
            p = point.view(-1, 3, 1)
        else:
            p = point.data.view(-1, 3, 1)

        temp = p - self[:, :, 3:]
        ret = Point3(data=(self[:, :, :3].transpose(1, 2) @ temp).view(-1, 3))

        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            # Right jacobians for SE(3) are computed
            Jg = torch.zeros(batch_size, 3, 6, dtype=self.dtype, device=self.device)
            Jg[:, 0, 0] = -1
            Jg[:, 1, 1] = -1
            Jg[:, 2, 2] = -1
            Jg[:, 0, 4] = -ret[:, 2]
            Jg[:, 1, 3] = ret[:, 2]
            Jg[:, 0, 5] = ret[:, 1]
            Jg[:, 2, 3] = -ret[:, 1]
            Jg[:, 1, 5] = -ret[:, 0]
            Jg[:, 2, 4] = ret[:, 0]
            # Jacobians for point
            Jpnt = self[:, :, :3].transpose(1, 2).expand(batch_size, 3, 3)

            jacobians.extend([Jg, Jpnt])

        return ret


rand_se3 = SE3.rand
randn_se3 = SE3.randn
