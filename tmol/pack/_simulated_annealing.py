import torch

from tmol.pack import PackerEnergyTables
from tmol.types import Tensor


def run_simulated_annealing(
    energy_tables: PackerEnergyTables,
    pose_seeds: torch.Tensor | None = None,
) -> tuple[Tensor[torch.float32][:, :], Tensor[torch.int32][:, :, :]]:
    """Rank rotamer assignments with GPU simulated annealing.

    Args:
        energy_tables: One- and two-body energies plus their rotamer layout.
        pose_seeds: Optional one-dimensional int64 tensor with one seed per
            pose, stored on the energy-table device.

    Returns:
        Scores shaped ``[n_poses, n_final_trajectories]`` and local rotamer
        assignments shaped ``[n_poses, n_final_trajectories, max_n_res]``.

    Notes:
        The annealer starts 500 high-temperature trajectories. The best 25%
        seed ten low-temperature trajectories each, and the best 25% of those
        are fully quenched, leaving 312 ranked assignments per pose.
    """
    from tmol.pack.compiled import pack_anneal

    if pose_seeds is None:
        pose_seeds = torch.empty(
            0, dtype=torch.int64, device=energy_tables.energy1b.device
        )
    else:
        if pose_seeds.dtype != torch.int64:
            raise TypeError("pose_seeds must have dtype torch.int64.")
        if pose_seeds.device != energy_tables.energy1b.device:
            raise ValueError("pose_seeds must be on the energy-table device.")
        if pose_seeds.ndim != 1:
            raise ValueError("pose_seeds must be one-dimensional.")
        if pose_seeds.numel() != energy_tables.pose_n_res.numel():
            raise ValueError("pose_seeds must contain one seed per pose.")

    return pack_anneal(
        energy_tables.max_n_rotamers_per_pose,
        energy_tables.pose_n_res,
        energy_tables.pose_n_rotamers,
        energy_tables.pose_rotamer_offset,
        energy_tables.nrotamers_for_res,
        energy_tables.oneb_offsets,
        energy_tables.res_for_rot,
        energy_tables.chunk_size,
        energy_tables.chunk_offset_offsets,
        energy_tables.chunk_offsets,
        energy_tables.energy1b,
        energy_tables.energy2b,
        pose_seeds.contiguous(),
    )
