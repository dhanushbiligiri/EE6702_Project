import random
import numpy as np
from tqdm import tqdm
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def ensure_psd(mat: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    mat = symmetrize(mat)
    eigvals = np.linalg.eigvalsh(mat)
    min_eig = eigvals.min()
    if min_eig < jitter:
        mat = mat + np.eye(mat.shape[0]) * (jitter - min_eig + 1e-12)
    return mat


def stable_inv_and_logdet(mat: np.ndarray, jitter: float = 1e-8):
    mat = ensure_psd(mat, jitter=jitter)
    sign, logdet = np.linalg.slogdet(mat)
    if sign <= 0:
        mat = ensure_psd(mat, jitter=max(jitter, 1e-6))
        sign, logdet = np.linalg.slogdet(mat)
        if sign <= 0:
            raise np.linalg.LinAlgError("Matrix is not positive definite.")
    inv = np.linalg.inv(mat)
    return inv, logdet


def logsumexp_np(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    out = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out