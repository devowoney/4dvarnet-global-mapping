from collections import namedtuple
from pathlib import Path
import re
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytorch_lightning as pl
import toolz
import torch
import tqdm
import xarray as xr
import xrpatcher

_FDVPATH = '/Odyssey/private/j25lee/4dvarnet-global-mapping'
if _FDVPATH not in sys.path:
    sys.path.append(_FDVPATH)

PredictItem = namedtuple('PredictItem', ('input',))


class LitModel(pl.LightningModule):
    def __init__(
        self, patcher, model, norm_stats, save_dir, crop_val,
        out_dims=('time', 'lat', 'lon'), **kwargs,
    ):
        super().__init__()
        self.patcher = patcher
        self.solver = model
        self.norm_stats = norm_stats
        self.save_dir = Path(save_dir)
        self.save_dir.parent.mkdir(parents=True, exist_ok=True)
        self.out_dims = out_dims
        self.bs = 0
        self.kwargs = kwargs
        self.crop_val = crop_val
        self.mask_path = kwargs.get('mask_path')

    def predict_step(self, batch, batch_idx: int, *args, **kwargs):
        if batch_idx == 0:
            self.predict_data = []

        outputs = self.solver(batch)

        self.bs = self.bs or outputs.shape[0]
        m, s = self.norm_stats

        outputs = (outputs.cpu().numpy() * s + m)

        numitem = outputs.shape[0]
        num_devices = self.trainer.num_devices * self.trainer.num_nodes
        item_idxes = (
            batch_idx * self.bs + torch.arange(numitem)
        ) * num_devices + self.global_rank

        assert len(self.out_dims) == len(outputs[0].shape)

        for i, idx in enumerate(item_idxes):
            out = outputs[i]
            c = self.patcher[idx].coords.to_dataset()[list(self.out_dims)]
            da = xr.DataArray(out, dims=self.out_dims, coords=c.coords)
            self.predict_data.append(da.astype(np.float32))

    def on_predict_end(self):
        """
        Merge the patches
        """
        # Retrieve patch dimensions
        p = self.predict_data[0]
        time, lat, lon = p.time.shape[0], p.lat.shape[0], p.lon.shape[0]

        def _crop(x):
            return crop(x, crop_val=self.crop_val)

        weight = self.kwargs.get(
            'weight',
            build_weight(
                patch_dims=dict(time=time, lat=lat, lon=lon),
                dim_weights=dict(time=triang, lat=_crop, lon=_crop)
            ),
        )
        out_var = self.kwargs.get('out_var', 'sla')
        _cround = self.kwargs.get('_cround', dict(lat=3, lon=3))
        out_coords = self.kwargs['out_coords']

        ## TODO: actual stuff
        for c, nd in _cround.items():
            out_coords[c] = np.round(out_coords[c], nd)
        out_coords = xr.Dataset(coords=out_coords)
        dims_shape = dict(**out_coords.sizes)

        rec_da = xr.DataArray(
            np.zeros(list(dims_shape.values())),
            dims=list(dims_shape.keys()),
            coords=out_coords.coords,
        )

        count_da = xr.zeros_like(rec_da)
        n_batches = len(self.predict_data)
        for _ in tqdm.tqdm(range(n_batches)):
            da = self.predict_data.pop(0)
            da = da.assign_coords(
                **{c: np.round(da[c].values, nd) for c, nd in _cround.items()}
            )
            w = xr.zeros_like(da) + weight
            wda = da * w
            coords_labels = set(dims_shape.keys()).intersection(da.coords.dims)
            da_co = {c: da[c].values for c in coords_labels}
            rec_da.loc[da_co] = rec_da.sel(da_co) + wda
            count_da.loc[da_co] = count_da.sel(da_co) + w

        final_reconstruction = (rec_da / count_da).to_dataset(name=out_var)

        if self.kwargs.get('save_cropped'):
            final_reconstruction = final_reconstruction.isel(
                lat=slice(self.crop_val, -self.crop_val),
                lon=slice(self.crop_val, -self.crop_val),
            )

        if self.kwargs.get('output_dc_format'):
            print('>>> Format the output for datachallenge OSE global 2019...')

            final_reconstruction = (
                final_reconstruction
                .rename(lat='latitude', lon='longitude')
                .sel(longitude=slice(-180, 180))
                .isel(longitude=slice(None, -1))
            )

            if self.mask_path:
                mask = xr.open_dataset(self.mask_path).mask
                if 'lat' in mask.coords or 'lon' in mask.coords:
                    mask = mask.rename(lat='latitude', lon='longitude')
                mask = mask.sel(latitude=final_reconstruction.latitude)
            else:
                mask = np.ones_like(final_reconstruction[out_var])

            final_reconstruction = (
                final_reconstruction
                .where(mask)
            )

            final_reconstruction.coords['longitude'] = (
                (final_reconstruction.coords['longitude'] + 360) % 360
            )
            final_reconstruction = final_reconstruction.sortby('longitude')
            final_reconstruction.latitude.attrs['units'] = 'degrees_north'
            final_reconstruction.longitude.attrs['units'] = 'degrees_east'

        final_reconstruction.to_netcdf(self.save_dir)


class XrDataset(torch.utils.data.Dataset):
    def __init__(self, patcher, postpro_fns=(PredictItem._make,)):
        self.patcher = patcher
        self.postpro_fns = postpro_fns or [lambda x: x.values]

    def __getitem__(self, idx):
        item = self.patcher[idx].load()
        item = toolz.thread_first(item, *self.postpro_fns)
        return item

    def __len__(self):
        return len(self.patcher)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


@hydra.main(version_base=None)
def _run(cfg):
    """
    Infer and merge.

    Accepted configuration keys are: (« * » means mandatory)

    *checkpoint_path (str):
        path to the checkpoint of the model to be used

    *config_path (str): path to the configuration file used to
        generate the model

    *input (str):
        path to the input file. Reconstructed field will have the same
        dimensions

    *output_path (str):
        location where the reconstruction must be stored at

    *patch (str):
        dimensions of patches (must follow the pattern time-lat-lon)

    *stride (str):
        strides for the patching (must follow the pattern time-lat-lon)

    batch_size (int):
        number of patches per batch

    devices (int):
        number of GPU to be used

    input_var (str):
        name of the variable from the input file to use

    mask_path (str):
        Path to the land-water mask to apply at the end of the inference

    mean (float):
        mean value to which values will be centred around

    num_worker (int):
        number of CPU to use for the data loading. If not specified,
        will be computed on the tracks data

    output_var (str):
        name of the variable in the output file

    period_from, period_to (str, str):
        period bounds to extract from the input data

    save_cropped (boolean):
        if True, remove the crop from the final reconstruction file,
        otherwise, the final reconstruction file will contain the `nan`
        border

    std (float):
        standard deviation to which values will be rescaled by. If not
        specified, will be computed on the tracks data
    """
    _from, _to = cfg.get('period_from'), cfg.get('period_to')
    sel = dict(time=slice(_from, _to))

    trainer = pl.Trainer(
        inference_mode=False, accelerator='gpu', devices=cfg.get('devices', 1),
        enable_checkpointing=False, logger=False,
    )

    patcher = xrpatcher.XRDAPatcher(
        da=xr.open_dataset(cfg['input'])[cfg.get('input_var', 'ssh')].sel(sel),
        patches=extract_dim_from_str(cfg['patch']),
        strides=extract_dim_from_str(cfg['stride']),
        check_full_scan=True,
    )

    solver = load_from_cfg(cfg['config_path'], key='model')
    ckpt = torch.load(cfg['checkpoint_path'], weights_only=True)
    solver.load_state_dict(ckpt['state_dict'])

    mean, std = (
        cfg.get('mean', patcher.da.mean().item()),
        cfg.get('std', patcher.da.std().item()),
    )
    norm_stats = mean, std

    torch_ds = XrDataset(
        patcher=patcher,
        postpro_fns=(
            lambda item: PredictItem._make((item.values.astype(np.float32),)),
            lambda item: item._replace(input=(item.input - mean) / std),
        ),
    )

    dl = torch.utils.data.DataLoader(
        torch_ds, batch_size=cfg.get('batch_size', 128),
        num_workers=cfg.get('num_workers', 1)
    )

    resolution = (patcher.da.lat[1] - patcher.da.lat[0]).item()

    litmod = LitModel(
        patcher, solver, norm_stats, save_dir=cfg['output_path'],
        crop_val=int(1/resolution), save_cropped=cfg.get('save_cropped', True),
        out_var=cfg.get('output_var', 'ssh'),
        out_coords=dict(
            time=pd.date_range(
                patcher.da['time'][0].dt.date.item(),
                patcher.da['time'][-1].dt.date.item(),
                freq='1D',
            ),
            lat=np.arange(
                patcher.da.lat[0].item(),
                patcher.da.lat[-1].item() + resolution,
                resolution,
            ),
            lon=np.arange(
                patcher.da.lon[0].item(),
                patcher.da.lon[-1].item() + resolution,
                resolution,
            ),
        ),
        output_dc_format=cfg.get('output_dc_format', False),
        mask_path=cfg.get('mask_path', None)
    )
    trainer.predict(litmod, dataloaders=dl)


def build_weight(patch_dims, dim_weights=None):
    if not dim_weights:
        dim_weights = dict(time=triang, lat=crop, lon=crop)

    return (
        dim_weights.get('time', np.ones)(patch_dims['time'])[:, None, None]
        * dim_weights.get('lat', np.ones)(patch_dims['lat'])[None, :, None]
        * dim_weights.get('lon', np.ones)(patch_dims['lon'])[None, None, :]
    )


def crop(n, crop_val):
    if crop_val:
        w = np.zeros(n)
        w[crop_val:-crop_val] = 1.0
        return w
    return np.ones(n)


def extract_dim_from_str(string):
    """
    Extract dimensions from a "A-B-C" string and return a dictionary
    containing time=A, lat=B and lon=C.
    """
    a = re.match(r'(\d+)-(\d+)-(\d+)', string)
    return dict(
        time=int(a.group(1)), lat=int(a.group(2)), lon=int(a.group(3))
    )


def load_from_cfg(cfg_path, key):
    """
    Load configurations from a specified file and instantiate the
    desired node.
    """
    cfg = OmegaConf.load(Path(cfg_path))
    node = OmegaConf.select(cfg, key)
    return hydra.utils.call(node)


def triang(n, min=0.05):
    return np.clip(1 - np.abs(np.linspace(-1, 1, n)), min, 1.0)


if __name__ == "__main__":
    _run()
