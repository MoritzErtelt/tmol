import torch

from tmol.pack.datatypes import PackerEnergyTables


def run_simulated_annealing(
    energy_tables: PackerEnergyTables, pose_seeds: torch.Tensor | None = None
):
    """Run GPU simulated annealing.

    Phase 1 (hi-temp SA): 500 trajectories run at high temperature
    Phase 2 (lo-temp SA): Each top hi-temp trajectory seeds 10 lo-temp
    trajectories, then round1_cut = 0.25 keeps the top 25%
      -> 500 * 10 * 0.25 = 1250 trajectories
    Phase 3 (full quench): round2_cut = 0.25 keeps the top 25% of those
      -> int(1250 * 0.25) = 312 trajectories
    """
    from tmol.pack.compiled.compiled import pack_anneal

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
