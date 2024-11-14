import numpy as np
from typing import Union, List

def chop_and_shift(paths: np.ndarray, chops: Union[int, List[int]]):
    """
    path: collection of paths of shape (n_samples, length, dim)
    chops: points at which to chop the path, if int needs to divide length - 1 and will be broadcasted
    return:
    chopped_paths: 
        if chops is int: a np.ndarray of shape (n_samples, (length - 1) // chops, chops + 1, dim)
        else: a list of length len(chops) - 1 containing the chopped paths with shapes (n_samples, chops[i+1] + 1 - chops[i], dim)
    """
    length = paths.shape[-2]
    if isinstance(chops, int):
        assert length % chops == 1
        return np.stack(
            chop_and_shift(
                paths, 
                chops = [i*chops for i in range((length - 1) // chops + 1)]
                ),
            axis=-3
            )
    else:
        chopped_paths = []
        for N_start, N_end in zip(chops[:-1], chops[1:]):
            chopped_paths.append(paths[..., N_start:(N_end+1), :] - paths[..., N_start:N_start+1, :])
        return chopped_paths