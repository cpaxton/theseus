import torch
import math
import theseus as th

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)


# returns a uniformly random point of the 2-sphere
def random_S2():
    theta = torch.rand(()) * math.tau
    z = torch.rand(()) * 2 - 1
    r = torch.sqrt(1 - z ** 2)
    return torch.tensor([r * torch.cos(theta), r * torch.sin(theta), z]).double()


# returns a uniformly random point of the 3-sphere
def random_S3():
    u, v, w = torch.rand(3)
    return torch.tensor(
        [
            torch.sqrt(1 - u) * torch.sin(math.tau * v),
            torch.sqrt(1 - u) * torch.cos(math.tau * v),
            torch.sqrt(u) * torch.sin(math.tau * w),
            torch.sqrt(u) * torch.cos(math.tau * w),
        ]
    ).double()


def randomSmallQuaternion(max_degrees, min_degrees=0):
    x, y, z = random_S2()
    theta = (
        (min_degrees + (max_degrees - min_degrees) * torch.rand(())) * math.tau / 360.0
    )
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.tensor([c, s * x, s * y, s * z])


from typing import Optional, Tuple, List


def softLossCauchy(x, radius):
    ratio = (x + radius) / radius
    val = torch.log(ratio) * radius
    der = 1.0 / ratio
    return val, der


def softLossHuberLike(x, radius):
    ratio = (x + radius) / radius
    sq = torch.sqrt(ratio)
    val = (sq - 1) * radius
    der = 0.5 / sq
    return val, der


softLoss = softLossHuberLike


class ReprojErr(th.CostFunction):
    def __init__(
        self,
        camRot: th.SO3,
        camTr: th.Point3,
        lossRadius: th.Vector,
        focalLength: th.Vector,
        worldPoint: th.Vector,
        imageFeaturePoint: th.Vector,
        i: int,
        name: Optional[str] = None,
    ):
        super().__init__(
            cost_weight=th.ScaleCostWeight(
                th.Vector(
                    data=torch.tensor([1.0], dtype=torch.float64), name=f"weight_{i}"
                )
            ),
            name=name,
        )
        self.camRot = camRot
        self.camTr = camTr
        self.lossRadius = lossRadius
        self.focalLength = focalLength
        self.worldPoint = worldPoint
        self.imageFeaturePoint = imageFeaturePoint

        self.register_optim_vars(["camRot", "camTr"])
        self.register_aux_vars(
            ["lossRadius", "focalLength", "imageFeaturePoint", "worldPoint"]
        )

    def error(self) -> torch.Tensor:
        camObsPoint = self.camRot.rotate(self.worldPoint) + self.camTr
        projObsPoint = camObsPoint[:, :2] / camObsPoint[:, 2:3] * self.focalLength.data
        err = projObsPoint - self.imageFeaturePoint.data
        # return err # no lossRadius

        errNorm = torch.norm(err, dim=1).unsqueeze(1)
        expLoss = torch.exp(lossRadius.data)

        val, der = softLoss(errNorm, expLoss)
        return val

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        camObsPoint = self.camRot.rotate(self.worldPoint) + self.camTr
        batch_size = self.camRot.shape[0]
        X = torch.zeros((batch_size, 3, 3), dtype=torch.float64)
        rotApplVec = self.worldPoint
        X[:, 0, 1] = rotApplVec[:, 2]
        X[:, 0, 2] = -rotApplVec[:, 1]
        X[:, 1, 0] = -rotApplVec[:, 2]
        X[:, 1, 2] = rotApplVec[:, 0]
        X[:, 2, 0] = rotApplVec[:, 1]
        X[:, 2, 1] = -rotApplVec[:, 0]
        J = torch.cat(
            (
                torch.bmm(self.camRot.data, X),
                torch.eye(3, 3).unsqueeze(0).repeat(batch_size, 1, 1),
            ),
            dim=2,
        )
        # J = torch.zeros((self.camRot.shape[0], 3, 6), dtype=torch.float64)
        # J[:, :, 3:6] = torch.eye(3, 3) # d/dTr
        # J[:, :, :3] = torch.bmm(self.camRot.data, X)

        projObsPoint = (camObsPoint[:, :2] / camObsPoint[:, 2:]) * self.focalLength.data
        dNum = J[:, 0:2, :]
        NumDDen_Den = torch.bmm(
            camObsPoint[:, :2].unsqueeze(2),
            (J[:, 2, :] / camObsPoint[:, 2:3]).unsqueeze(1),
        )
        Dproj = (
            (dNum - NumDDen_Den)
            / camObsPoint[:, 2:].unsqueeze(2)
            * self.focalLength.data.unsqueeze(2)
        )
        err = projObsPoint - self.imageFeaturePoint.data
        # return [Dproj[:, :, :3], Dproj[:, :, 3:]], err # no lossRadius

        errNorm = torch.norm(err, dim=1).unsqueeze(1)
        errDir = err / errNorm
        normJac = torch.bmm(errDir.unsqueeze(1), Dproj)
        expLoss = torch.exp(lossRadius.data)

        val, der = softLoss(errNorm, expLoss)
        softJac = normJac * der.unsqueeze(1)

        return [softJac[:, :, :3], softJac[:, :, 3:]], val

        # retv = (torch.log(errNorm / expLoss + 1) * expLoss) ###.unsqueeze(1)
        # retvJac = normJac * (expLoss / (errNorm + expLoss)).unsqueeze(1)
        # outs = [retvJac[:, :, :3], retvJac[:, :, 3:]], retv # no lossRadius
        # print([retvJac[:, :, :3], retvJac[:, :, 3:]], retv)
        # print([retvJac[:, :, :3].shape, retvJac[:, :, 3:].shape], retv.shape)
        # outs

    def dim(self) -> int:
        return 2

    # calls to() on the cost weight, variables and any internal tensors
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self):
        return ReprojErr(
            camRot,
            camTr,
            lossRadius,
            focalLength,
            worldPoint,
            imageFeaturePoint,
            i,
            name,
        )


if False:
    camRot = th.SO3(
        torch.cat(
            [randomSmallQuaternion(max_degrees=20).unsqueeze(0) for _ in range(4)]
        ),
        name="camRot",
    )
    camTr = th.Point3(data=torch.zeros((4, 3), dtype=torch.float64), name="camTr")
    camTr.data[:, 2] += 5.0
    focalLenght = th.Vector(
        data=torch.tensor([1000], dtype=torch.float64).repeat(4).unsqueeze(1),
        name="focalLength",
    )
    lossRadius = th.Vector(
        data=torch.tensor([0], dtype=torch.float64).repeat(4).unsqueeze(1),
        name="lossRadius",
    )
    worldPoint = th.Vector(
        data=torch.rand((4, 3), dtype=torch.float64), name="worldPoint"
    )
    camPoint = camRot.rotate(worldPoint) + camTr
    imageFeaturePoint = th.Vector(
        data=camPoint[:, :2] / camPoint[:, 2:] + torch.rand((4, 2)) * 50,
        name="imageFeaturePoint",
    )
    r = ReprojErr(
        camRot=camRot,
        camTr=camTr,
        focalLength=focalLenght,
        lossRadius=lossRadius,
        worldPoint=worldPoint,
        imageFeaturePoint=imageFeaturePoint,
        i=0,
    )

    r.lossRadius.data.requires_grad = True
    print(r.jacobians())
    (rotJac, trJac), err = r.jacobians()
    # r.error().backward(torch.tensor([1,2,3,4]).unsqueeze(1))
    # rotJac.backward(torch.rand(rotJac.shape))
    trJac.backward(torch.rand(trJac.shape))
    r.lossRadius.data.grad


# unit test for Cost term
camRot = th.SO3(
    torch.cat([randomSmallQuaternion(max_degrees=20).unsqueeze(0) for _ in range(4)]),
    name="camRot",
)
camTr = th.Point3(data=torch.zeros((4, 3), dtype=torch.float64), name="camTr")
camTr.data[:, 2] += 5.0
focalLenght = th.Vector(
    data=torch.tensor([1000], dtype=torch.float64).repeat(4).unsqueeze(1),
    name="focalLength",
)
lossRadius = th.Vector(
    data=torch.tensor([0], dtype=torch.float64).repeat(4).unsqueeze(1),
    name="lossRadius",
)
worldPoint = th.Vector(data=torch.rand((4, 3), dtype=torch.float64), name="worldPoint")
camPoint = camRot.rotate(worldPoint) + camTr
imageFeaturePoint = th.Vector(
    data=camPoint[:, :2] / camPoint[:, 2:] + torch.rand((4, 2)) * 50,
    name="imageFeaturePoint",
)
r = ReprojErr(
    camRot=camRot,
    camTr=camTr,
    focalLength=focalLenght,
    lossRadius=lossRadius,
    worldPoint=worldPoint,
    imageFeaturePoint=imageFeaturePoint,
    i=0,
)

baseVal = r.error()
baseCamRot = r.camRot.copy()
baseCamTr = r.camTr.copy()
nErr = baseVal.shape[1]
nJac = torch.zeros((r.camRot.data.shape[0], nErr, 6), dtype=torch.float64)
epsilon = 1e-8
for i in range(6):
    if i >= 3:
        r.camTr = baseCamTr.copy()
        r.camTr.data[:, i - 3] += epsilon
        r.camRot = baseCamRot.copy()
    else:
        r.camTr = baseCamTr.copy()
        v = torch.zeros((r.camRot.data.shape[0], 3), dtype=torch.float64)
        v[:, i] += epsilon
        r.camRot = baseCamRot.retract(v)
    pertVal = r.error()
    nJac[:, :, i] = (pertVal - baseVal) / epsilon

rotNumJac = nJac[:, :, :3]
trNumJac = nJac[:, :, 3:]

(rotJac, trJac), _ = r.jacobians()

print(
    "|numJac-analiticJac|: ",
    float(torch.norm(rotNumJac - rotJac)),
    float(torch.norm(trNumJac - trJac)),
)


def add_noise_and_outliers(
    projPoints,
    noiseSize=1,
    noiseLinear=True,
    proportionOutliers=0.05,
    outlierDistance=500,
):

    if noiseLinear:
        featImagePoints = projPoints + noiseSize * (
            torch.rand(projPoints.shape, dtype=torch.float64) * 2 - 1
        )
    else:  # normal, stdDev = noiseSize
        featImagePoints = projPoints + torch.normal(
            mean=torch.zeros(projPoints.shape), std=noiseSize, dtype=torch.float64
        )

    # add real bad outliers
    outliersMask = torch.rand(featImagePoints.shape[0]) < proportionOutliers
    numOutliers = featImagePoints[outliersMask].shape[0]
    featImagePoints[outliersMask] += outlierDistance * (
        torch.rand((numOutliers, projPoints.shape[1]), dtype=projPoints.dtype) * 2 - 1
    )
    return featImagePoints


class LocalizationSample:
    def __init__(self, num_points=60, focalLength=1000):
        self.focalLength = th.Variable(
            data=torch.tensor([focalLength], dtype=torch.float64), name="focalLength"
        )

        # pts = [+/-10, +/-10, +/-1]
        self.worldPoints = torch.cat(
            [
                torch.rand(2, num_points, dtype=torch.float64) * 20 - 10,
                torch.rand(1, num_points, dtype=torch.float64) * 2 - 1,
            ]
        ).T

        # gtCamPos = [+/-3, +/-3, 5 +/-1]
        gtCamPos = th.Point3(
            torch.tensor(
                [
                    [
                        torch.rand((), dtype=torch.float64) * 3,
                        torch.rand((), dtype=torch.float64) * 3,
                        5 + torch.rand((), dtype=torch.float64),
                    ]
                ]
            ),
            name="gtCamPos",
        )
        self.gtCamRot = th.SO3(randomSmallQuaternion(max_degrees=20), name="gtCamRot")
        self.gtCamTr = (-self.gtCamRot.rotate(gtCamPos)).copy(new_name="gtCamTr")

        camPoints = self.gtCamRot.rotate(self.worldPoints) + self.gtCamTr
        projPoints = camPoints[:, :2] / camPoints[:, 2:3] * self.focalLength.data
        self.imageFeaturePoints = add_noise_and_outliers(projPoints)

        smallRot = th.SO3(randomSmallQuaternion(max_degrees=0.3))
        smallTr = torch.rand(3, dtype=torch.float64) * 0.1
        self.obsCamRot = smallRot.compose(self.gtCamRot).copy(new_name="obsCamRot")
        self.obsCamTr = (smallRot.rotate(self.gtCamTr) + smallTr).copy(
            new_name="obsCamTr"
        )


l = LocalizationSample()


# create optimization problem
camRot = l.obsCamRot.copy(new_name="camRot")
camTr = l.obsCamTr.copy(new_name="camTr")
lossRadius = th.Vector(1, name="lossRadius", dtype=torch.float64)
focalLength = th.Vector(1, name="focalLength", dtype=torch.float64)

# NOTE: if not set explicitly will crash using a weight of wrong type `float32`
weight = th.ScaleCostWeight(
    th.Vector(data=torch.tensor([1.0], dtype=torch.float64), name="weight")
)

# Set up objective
objective = th.Objective(dtype=torch.float64)
for i in range(len(l.worldPoints)):
    worldPoint = th.Vector(data=l.worldPoints[i], name=f"worldPoint_{i}")
    imageFeaturePoint = th.Vector(
        data=l.imageFeaturePoints[i], name=f"imageFeaturePoint_{i}"
    )

    # optim_vars = [camRot, camTr]
    # aux_vars = [lossRadius, focalLength, worldPoint, imageFeaturePoint]
    cost_function = ReprojErr(
        camRot=camRot,
        camTr=camTr,
        focalLength=focalLength,
        lossRadius=lossRadius,
        worldPoint=worldPoint,
        imageFeaturePoint=imageFeaturePoint,
        i=i,
    )
    objective.add(cost_function)

# Create optimizer
optimizer = th.LevenbergMarquardt(  # GaussNewton(
    objective,
    max_iterations=10,
    step_size=0.3,
)

# Set up Theseus layer
theseus_optim = th.TheseusLayer(optimizer)


# Create dataset
# NOTE: composition of SO3 rotations is often not a valid rotation (.copy fails)
loc_samples = [LocalizationSample() for _ in range(16)]
batch_size = 4
num_batches = (len(loc_samples) + batch_size - 1) // batch_size


def get_batch(b):
    assert b * batch_size < len(loc_samples)
    batch_ls = loc_samples[b * batch_size : (b + 1) * batch_size]
    batch_data = {
        "camRot": th.SO3(data=torch.cat([l.obsCamRot.data for l in batch_ls])),
        "camTr": th.Point3(data=torch.cat([l.obsCamTr.data for l in batch_ls])),
        "focalLength": th.Vector(
            data=torch.cat([l.focalLength.data.unsqueeze(1) for l in batch_ls]),
            name="focalLength",
        ),
    }

    # batch of 3d points and 2d feature points
    for i in range(len(batch_ls[0].worldPoints)):
        batch_data[f"worldPoint_{i}"] = th.Vector(
            data=torch.cat([l.worldPoints[i : i + 1].data for l in batch_ls]),
            name=f"worldPoint_{i}",
        )
        batch_data[f"imageFeaturePoint_{i}"] = th.Vector(
            data=torch.cat([l.imageFeaturePoints[i : i + 1].data for l in batch_ls]),
            name=f"imageFeaturePoint_{i}",
        )

    gtCamRot = th.SO3(data=torch.cat([l.gtCamRot.data for l in batch_ls]))
    gtCamTr = th.Point3(data=torch.cat([l.gtCamTr.data for l in batch_ls]))
    return batch_data, gtCamRot, gtCamTr


# Outer optimization loop
lossRadius_tensor = torch.nn.Parameter(torch.tensor([-3], dtype=torch.float64))
model_optimizer = torch.optim.Adam([lossRadius_tensor], lr=1.0)

# print(f"Initial a value: {a_tensor.item()}")

num_epochs = 100

camRotVar = theseus_optim.objective.optim_vars["camRot"]
camTrVar = theseus_optim.objective.optim_vars["camTr"]
for epoch in range(num_epochs):
    print(" ******************* EPOCH {epoch} ******************* ")
    epoch_loss = 0.0
    epoch_b = []  # keep track of the current b values for each model in this epoch
    for i in range(num_batches):
        print(f"BATCH {i}/{num_batches}")
        model_optimizer.zero_grad()
        theseus_inputs, gtCamRot, gtCamTr = get_batch(i)
        theseus_inputs["lossRadius"] = lossRadius_tensor.repeat(
            gtCamTr.data.shape[0]
        ).unsqueeze(1)

        theseus_outputs, info = theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={"verbose": False}
        )

        cam_rot_loss = th.local(camRotVar, gtCamRot).norm(dim=1)
        cam_tr_loss = th.local(camTrVar, gtCamTr).norm(dim=1, p=1)
        loss = (100 * cam_rot_loss + cam_tr_loss)
        loss = torch.where(loss < 10e5, loss, 0.0).sum()
        loss.backward()
        model_optimizer.step()

        loss_value = torch.sum(loss.detach()).item()
        epoch_loss += loss_value

    print(
        f"Epoch: {epoch} Loss: {epoch_loss} "
        f"Kernel Radius: exp({lossRadius_tensor.data.item()})="
        f"{torch.exp(lossRadius_tensor.data).item()}"
    )
