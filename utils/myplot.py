import os
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
# from mask_utils import tile_mask

def savefig(plot, fname, ftype='pdf', fontsize=10, dpi=250):
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'{fname}_{timestr}.{ftype}', bbox_inches='tight', dpi=dpi)
    plt.savefig(f'{fname}_latest.{ftype}', bbox_inches='tight', dpi=dpi)

# def visualize_heat(image, heat, path, tile_size, overwrite=True, tile=True):
#     if os.path.exists(path) and not overwrite:
#         return
    
#     fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
#     if tile:
#         heat = tile_mask(heat, tile_size)[0, 0, :, :]
#     else:
#         heat = heat[0, 0, :, :]
#     ax = sns.heatmap(
#         heat.cpu().detach().numpy(),
#         zorder=3,
#         alpha=0.5,
#         ax=ax,
#         xticklabels=False,
#         yticklabels=False,
#     )
#     ax.imshow(image, zorder=3, alpha=0.5)
#     ax.tick_params(left=False, bottom=False)
#     # Path(path).parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, bbox_inches="tight")
#     plt.close(fig)