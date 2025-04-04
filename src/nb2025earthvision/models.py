import pathlib
from datetime import datetime
from typing import Protocol
import torch
import torch.nn as nn
import sarssm

import nb2025earthvision as ev25
from nb2025earthvision import constants


class MoisturePredictor(Protocol):
    def predict_soil_moisture(self, t3, incidence) -> torch.Tensor: ...


# Neural encoder


class CoherencyEncoder(nn.Module):
    """
    Coherency matrix encoder: transform coherency matrices to the latent parameter space.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 20)
        self.activation1 = nn.LeakyReLU(negative_slope=0.05)
        self.linear2 = nn.Linear(20, 40)
        self.activation2 = nn.LeakyReLU(negative_slope=0.05)
        self.linear3 = nn.Linear(40, 20)
        self.activation3 = nn.LeakyReLU(negative_slope=0.05)
        # outputs to the latent space
        self.out_m_d = nn.Linear(20, 1)
        self.out_m_v = nn.Linear(20, 1)
        self.out_soil_mst = nn.Linear(20, 1)
        self.out_delta = nn.Linear(20, 1)

    def feature_transform_v1(self, t3_batch, incidence_batch):
        """
        Simple feature transform:
        - pauli powers (T3 diagonal)
        - real and imaginary parts of T[0,1]
        - incidence angle
        t3_batch.shape = (*B, 3, 3)
        incidence_batch.shape = (*B,)
        """
        pauli1 = t3_batch[..., 0, 0].real
        pauli2 = t3_batch[..., 1, 1].real
        pauli3 = t3_batch[..., 2, 2].real
        offdiag12r = t3_batch[..., 0, 1].real
        offdiag12i = t3_batch[..., 0, 1].imag
        stacked = torch.stack((pauli1, pauli2, pauli3, offdiag12r, offdiag12i, incidence_batch), dim=-1)
        return stacked

    def feature_transform_v2(self, t3_batch, incidence_batch):
        """
        Feature transform:
        - T3 diagonal: log of pauli powers + 1
        - normalized T[0,1]: real and imaginary parts of Pauli1-2 coherence
        - incidence angle
        t3_batch.shape = (*B, 3, 3)
        incidence_batch.shape = (*B,)
        """
        pauli1 = t3_batch[..., 0, 0].real
        pauli2 = t3_batch[..., 1, 1].real
        pauli3 = t3_batch[..., 2, 2].real
        offdiag12r = t3_batch[..., 0, 1].real
        offdiag12i = t3_batch[..., 0, 1].imag
        pauli12_norm = torch.sqrt(pauli1 * pauli2)
        pauli1 = torch.log(pauli1 + 1)
        pauli2 = torch.log(pauli2 + 1)
        pauli3 = torch.log(pauli3 + 1)
        offdiag12r = offdiag12r / pauli12_norm
        offdiag12i = offdiag12i / pauli12_norm
        stacked = torch.stack((pauli1, pauli2, pauli3, offdiag12r, offdiag12i, incidence_batch), dim=-1)
        return stacked

    def _constrain_values(self, values, vmin, vmax):
        return nn.functional.sigmoid(values) * (vmax - vmin) + vmin

    def forward(self, t3_batch, incidence_batch):
        x = self.feature_transform_v2(t3_batch, incidence_batch)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        # physical parameters
        m_d = self.out_m_d(x).squeeze(-1)
        m_v = self.out_m_v(x).squeeze(-1)
        soil_mst = self._constrain_values(
            self.out_soil_mst(x).squeeze(-1), constants.soil_mst_min, constants.soil_mst_max
        )
        delta = self._constrain_values(self.out_delta(x).squeeze(-1), constants.delta_min, constants.delta_max)
        return m_d, m_v, soil_mst, delta

    def predict_soil_moisture(self, t3_batch, incidence_batch) -> torch.Tensor:
        m_d, m_v, soil_mst, delta = self.forward(t3_batch, incidence_batch)
        return soil_mst


def save_encoder(encoder: CoherencyEncoder, path: pathlib.Path):
    torch.save(encoder.state_dict(), path)


def load_encoder(path: pathlib.Path):
    encoder = CoherencyEncoder()
    encoder.load_state_dict(torch.load(path, weights_only=True))
    return encoder


# Physical model


def xbragg(m_s, eps_s, delta, incidence):
    """
    Forward X-Bragg model, scattering from a slightly rough surface.
    All inputs must be broadcastable to each other resulting in batch_shape.
    The output is a batch of matrices of shape (*batch_shape, 3, 3).

    Reference: "Potential of Estimating Soil Moisture under Vegetation Cover by Means of PolSAR"
    """
    batch_shape = torch.broadcast_shapes(m_s.shape, eps_s.shape, delta.shape, incidence.shape)
    theta_s_sin_2 = torch.sin(incidence) ** 2
    theta_s_cos = torch.cos(incidence)
    # bragg terms
    sqrt = torch.sqrt(eps_s - theta_s_sin_2)
    r_h = (theta_s_cos - sqrt) / (theta_s_cos + sqrt)
    r_v_up = (eps_s - 1) * (theta_s_sin_2 - eps_s * (1 + theta_s_sin_2))
    r_v_down = (eps_s * theta_s_cos + sqrt) ** 2
    r_v = r_v_up / r_v_down
    beta = (r_h - r_v) / (r_h + r_v)
    f_s = (m_s**2) / 2 * torch.abs(r_h + r_v) ** 2
    # x-bragg terms, torch sinc is `sin(pi x)/(pi x)` but paper is `sin(x)/x`
    sinc2 = torch.sinc(2 / torch.pi * delta)
    sinc4 = torch.sinc(4 / torch.pi * delta)
    beta_sinc2 = beta * sinc2
    half_beta_sqr = 0.5 * torch.abs(beta) ** 2
    # building the matrix
    m_xbragg = torch.zeros((*batch_shape, 3, 3), dtype=torch.cfloat)
    m_xbragg[..., 0, 0] = 1
    m_xbragg[..., 0, 1] = torch.conj(beta_sinc2)
    m_xbragg[..., 1, 0] = beta_sinc2
    m_xbragg[..., 1, 1] = half_beta_sqr * (1 + sinc4)
    m_xbragg[..., 2, 2] = half_beta_sqr * (1 - sinc4)
    return f_s[..., None, None] * m_xbragg  # shape (*batch_shape, 3, 3)


def dihedral(m_d, eps_s, eps_d, phi, incidence):
    """
    Forward dihedral model, scattering from two planes with different dielectrics.
    All inputs must be broadcastable to each other resulting in batch_shape.
    The output is a batch of matrices of shape (*batch_shape, 3, 3).
    """
    batch_shape = torch.broadcast_shapes(m_d.shape, eps_s.shape, eps_d.shape, phi.shape, incidence.shape)
    theta_s_sin_2 = torch.sin(incidence) ** 2
    theta_s_cos = torch.cos(incidence)
    theta_d_sin_2 = torch.sin(torch.pi / 2 - incidence) ** 2
    theta_d_cos = torch.cos(torch.pi / 2 - incidence)
    # dihedral terms
    sqrt_s = torch.sqrt(eps_s - theta_s_sin_2)
    sqrt_d = torch.sqrt(eps_d - theta_d_sin_2)
    r_h_s = (theta_s_cos - sqrt_s) / (theta_s_cos + sqrt_s)
    r_h_d = (theta_d_cos - sqrt_d) / (theta_d_cos + sqrt_d)
    r_v_s = (eps_s * theta_s_cos - sqrt_s) / (eps_s * theta_s_cos + sqrt_s)
    r_v_d = (eps_d * theta_d_cos - sqrt_d) / (eps_d * theta_d_cos + sqrt_d)
    phase = torch.exp(1j * phi)
    r_h_sd = r_h_s * r_h_d
    r_v_sd = r_v_s * r_v_d * phase
    f_d = (m_d**2 / 2) * torch.abs(r_h_sd + r_v_sd) ** 2
    dihedral_alpha = (r_h_sd - r_v_sd) / (r_h_sd + r_v_sd)
    # building the matrix
    m_dihedral = torch.zeros((*batch_shape, 3, 3), dtype=torch.cfloat)
    dihedral_alpha_abs_sqr = torch.abs(dihedral_alpha) ** 2
    m_dihedral[..., 0, 0] = dihedral_alpha_abs_sqr
    m_dihedral[..., 0, 1] = dihedral_alpha
    m_dihedral[..., 1, 0] = torch.conj(dihedral_alpha)
    m_dihedral[..., 1, 1] = 1
    return f_d[..., None, None] * m_dihedral  # shape (*batch_shape, 3, 3)


def random_volume():
    return (1 / 4) * torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )


class PhysicalDecoder(nn.Module):
    """
    Physical model decoder: transform parameters to coherency matrices.
    """

    def __init__(self):
        super().__init__()

    def forward_components(self, m_s, m_d, m_v, soil_mst, plant_mst, delta, phi, incidence, sand, clay, frequency):
        # get batch shape
        params: list[torch.Tensor] = [m_s, m_d, m_v, soil_mst, plant_mst, delta, phi, incidence, sand, clay, frequency]
        param_shapes = [p.shape for p in params]
        batch_shape = torch.broadcast_shapes(*param_shapes)
        # compute components
        eps_s = sarssm.moisture_to_eps_hallikainen(soil_mst, sand, clay, frequency)
        eps_d = sarssm.corn_moisture_to_eps_ulaby(plant_mst, frequency)
        xb = xbragg(m_s, eps_s, delta, incidence)
        dh = dihedral(m_d, eps_s, eps_d, phi, incidence)
        assert m_v.shape == batch_shape
        f_v = (m_v**2 / 2)[..., None, None]  # shape (*B) -> (*B, 1, 1)
        vol = f_v * random_volume()  # shapes (*B, 1, 1) x (3, 3) -> (*B, 3, 3)
        return xb, dh, vol

    def forward(self, m_s, m_d, m_v, soil_mst, plant_mst, delta, phi, incidence, sand, clay, frequency):
        xb, dh, vol = self.forward_components(
            m_s, m_d, m_v, soil_mst, plant_mst, delta, phi, incidence, sand, clay, frequency
        )
        return xb + dh + vol

    def _batch_trace_real(self, x):
        return torch.diagonal(x.real, dim1=-2, dim2=-1).sum(-1)

    def get_component_powers_np(self, m_s, m_d, m_v, soil_mst, plant_mst, delta, phi, incidence, sand, clay, frequency):
        xb, dh, vol = self.forward_components(
            m_s, m_d, m_v, soil_mst, plant_mst, delta, phi, incidence, sand, clay, frequency
        )
        # get powers
        xbragg_power = self._batch_trace_real(xb).cpu().detach().numpy()
        dihedral_power = self._batch_trace_real(dh).cpu().detach().numpy()
        volume_power = self._batch_trace_real(vol).cpu().detach().numpy()
        return xbragg_power, dihedral_power, volume_power


# Direct physical model inversion with optimization


class PhysicalInversionParameters(nn.Module):
    """
    Parameters for direct model inversion.
    """

    def __init__(self, t3: torch.Tensor, init_seed=None):
        super().__init__()
        self.t3 = t3
        *batch_shape, u, v = t3.shape
        assert u == 3 and v == 3
        self.batch_shape = batch_shape
        if init_seed is not None:
            torch.manual_seed(init_seed)
        self.m_d = self._init_param(batch_shape, 0.01, 0.1)
        self.m_v = self._init_param(batch_shape, 0.01, 0.1)
        self.soil_mst = self._init_param(batch_shape, constants.soil_mst_min, constants.soil_mst_max)
        self.delta = self._init_param(batch_shape, constants.delta_min, constants.delta_max)
        self.loss_history = []  # loss by fitting iteration

    def _init_param(self, shape, vmin, vmax):
        return nn.Parameter(torch.rand(shape, dtype=torch.float) * (vmax - vmin) + vmin)


class PhysicalInversionModel:
    """
    Direct physical model inversion with optimization.
    """

    def __init__(self, m_s, plant_mst, phi, sand, clay, frequency, seed=None):
        # calibrated parameters
        self.m_s = m_s
        self.plant_mst = plant_mst
        self.phi = phi
        # constant parameters requred for soil-moisture-to-dielectrics conversion
        self.sand = sand
        self.clay = clay
        self.frequency = frequency
        # optional seed for initialization
        self.seed = seed

    def invert_parameters(self, t3, incidence, verbose=False):
        # inversion hyperparameters
        iterations = 700
        lr = 0.3
        scheduler_step_size = 400
        scheduler_gamma = 0.1
        # inversion setup
        inverted_params = PhysicalInversionParameters(t3, self.seed)
        decoder = PhysicalDecoder()
        param_list = [inverted_params.m_d, inverted_params.m_v, inverted_params.soil_mst, inverted_params.delta]
        optimizer = torch.optim.Adam(param_list, lr=lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        start = datetime.now()
        for i in range(iterations):
            optimizer.zero_grad()
            reconstr = decoder.forward(
                m_s=self.m_s,
                m_d=inverted_params.m_d,
                m_v=inverted_params.m_v,
                soil_mst=inverted_params.soil_mst,
                plant_mst=self.plant_mst,
                delta=inverted_params.delta,
                phi=self.phi,
                incidence=incidence,
                sand=self.sand,
                clay=self.clay,
                frequency=self.frequency,
            )
            loss = ev25.mean_squared_error_matrix(t3, reconstr)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                # ensure parameters stay in the valid range
                inverted_params.soil_mst.clamp_(min=constants.soil_mst_min, max=constants.soil_mst_max)
                inverted_params.delta.clamp_(min=constants.delta_min, max=constants.delta_max)
            inverted_params.loss_history.append(loss.item())
            if verbose and (i == 0 or i % 100 == 99 or i == self.iterations - 1):
                print(f"  Iteration {i+1:<5.0f}   Loss = {loss.item():.7f}")
        end = datetime.now()
        if verbose:
            print(f"  Completed {self.iterations} iterations in {(end - start).total_seconds():.1f} s")
        return inverted_params

    def predict_soil_moisture(self, t3, incidence) -> torch.Tensor:
        parameters = self.invert_parameters(t3, incidence)
        return parameters.soil_mst
