import torch
import numpy as np

def merge_state_subkeys(
    dataset,
    use_keys: list[str] | None = None,
    filtered_keys: list[str] | None = None,
) -> list[str]:
    """Merge observation.state.* sub-keys into a single observation.state key.

    Some datasets (e.g. robocasa) store each state component under a separate key
    like observation.state.robot0_eef_pos instead of a flat observation.state vector.
    Diffusion policy (and others) expect a single observation.state key, so we patch
    dataset.meta in-place and return the selected sub-key names for use in collate_fn.

    Args:
        dataset: LeRobotDataset whose meta will be patched in-place.
        use_keys: Whitelist of sub-key suffixes to include (e.g. ["robot0_eef_pos"]).
                  Only these keys will be merged. Mutually exclusive with filtered_keys.
        filtered_keys: Blacklist of sub-key suffixes to exclude (e.g. ["robot0_joint_pos_cos"]).
                       All other keys will be merged. Mutually exclusive with use_keys.

    Returns:
        Sorted list of full feature key names that were merged into observation.state.
    """
    if use_keys is not None and filtered_keys is not None:
        raise ValueError("use_keys and filtered_keys are mutually exclusive.")

    all_state_subkeys = sorted(
        [k for k in dataset.meta.features if k.startswith("observation.state.")]
    )
    if not all_state_subkeys:
        return []
    # ['min', 'max', 'mean', 'std', 'count', 'q01', 'q10', 'q50', 'q90', 'q99']
    stat_types = list(dataset.meta.stats[all_state_subkeys[0]].keys())

    if use_keys is not None:
        full_use_keys = {f"observation.state.{s}" for s in use_keys}
        state_subkeys = [k for k in all_state_subkeys if k in full_use_keys]
    elif filtered_keys is not None:
        full_filtered_keys = {f"observation.state.{s}" for s in filtered_keys}
        state_subkeys = [k for k in all_state_subkeys if k not in full_filtered_keys]
    else:
        state_subkeys = all_state_subkeys

    if not state_subkeys:
        return []

    total_dim = sum(dataset.meta.features[k]["shape"][0] for k in state_subkeys)

    # Remove all observation.state.* from features (both selected and unselected)
    for k in all_state_subkeys:
        dataset.meta.features.pop(k)
    dataset.meta.features["observation.state"] = {
        "dtype": "float32",
        "shape": (total_dim,),
        "names": None,
    }

    # Patch stats across selected sub-keys in sorted order.
    # Vector stats (mean, std, min, max, q*): concatenate across subkeys.
    # Scalar-like stats (count): same value for all subkeys → take first.
    vector_stat_types = {"mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"}
    for stat_type in stat_types:
        values = [
            dataset.meta.stats[k][stat_type]
            for k in state_subkeys
            if k in dataset.meta.stats and stat_type in dataset.meta.stats[k]
        ]
        if not values:
            continue
        tensors = [
            (torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.as_tensor(v)).float()
            for v in values
        ]
        if stat_type in vector_stat_types:
            merged = torch.cat(tensors, dim=0)
        else:
            merged = tensors[0]
        dataset.meta.stats.setdefault("observation.state", {})[stat_type] = merged
    for k in all_state_subkeys:
        dataset.meta.stats.pop(k, None)

    return state_subkeys


def slice_action_meta(dataset, n_dims: int) -> None:
    """Patch dataset.meta in-place to keep only the first n_dims of the action.

    Updates both features shape and stats (mean, std, min, max, q*) so that
    the policy and normalizer see the reduced action space.
    """
    feat = dataset.meta.features.get("action")
    if feat is None:
        raise KeyError("'action' not found in dataset.meta.features")
    original_dim = feat["shape"][0]
    if n_dims >= original_dim:
        return
    feat["shape"] = (n_dims,)

    vector_stat_types = {"mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"}
    action_stats = dataset.meta.stats.get("action", {})
    for stat_type, val in action_stats.items():
        if stat_type not in vector_stat_types:
            continue
        if isinstance(val, torch.Tensor):
            action_stats[stat_type] = val[:n_dims]
        elif isinstance(val, (list, np.ndarray)):
            action_stats[stat_type] = list(val)[:n_dims]


def make_concat_state_collate_fn(state_subkeys: list[str], action_keep_dims: int | None = None):
    """Return a collate_fn that concatenates observation.state.* tensors into observation.state.

    Optionally slices action to first action_keep_dims dimensions.
    """
    from torch.utils.data import default_collate

    def collate_fn(batch):
        collated = default_collate(batch)
        if state_subkeys:
            collated["observation.state"] = torch.cat(
                [collated.pop(k) for k in state_subkeys], dim=-1
            )
        if action_keep_dims is not None:
            collated["action"] = collated["action"][..., :action_keep_dims]
        return collated

    return collate_fn
