# GLOENS native-grid vertical section extractor

Produces one vertical section per name, variable, and lead as PNG and NetCDF. Each NetCDF contains longitude/latitude, distance along section (km), and original T-grid `(grid_y, grid_x)` indices.

Uses the GLOENS 2D native curvilinear grid and nearest native T cells, including land. Zonal and meridional sections preserve their coordinate; other sections follow a great circle. Supply the native T-mask (--mask-file) to mask land and below-bottom cells. Continental crossings remain in the section as missing values, not displaced ocean samples. This is a tracer transect, not a conservative transport integration.

## Install

```bash
pip install -r requirements.txt

```

Keep Class3 and physical_stability as sibling directories (for example under /home/cregnier/Github). Class3 automatically adds the neighbouring physical_stability/src directory to its Python import path. No installation of physical_stability or manual PYTHONPATH configuration is required. Its reader dependencies must still be available in your Python environment.

## Remote 3D ensemble Zarr

```bash
python gloens_sections.py \
  --store 'oceanbench-bucket/PATH/TO/ACTUAL/GLOENS_3D_TS.zarr' \
  --init-date 20260101 \
  --sections def_section.yaml --section OVIDE_CLIVAR_A25 \
  --variables thetao so --first-day 0 --last-day 3 \
  --ensemble mean \
  --mask-file /data/rd_exchange/cregnier/static/iORCA025_meshmask_v3.1_binary.nc \
  --output sections_out
```

The Zarr path is **a placeholder**: the physical_stability reader requires you to supply the true 3D T/S store. SSH-only `zos` cannot supply vertical sections.

To plot member 7 (index 7, **zero-based**): replace `--ensemble mean` by `--ensemble members --member 7`.

## Local precomputed GLOENS ensemble-mean `.npy`

```bash
python gloens_sections.py \
  --npy-dir /path/to/CLASS4_scores --grid-file /path/to/iORCA025_domcfg_v3.1.nc \
  --init-date 20260101 --first-day 0 --last-day 2 \
  --sections def_section.yaml --variables thetao so \
  --mask-file /path/to/iORCA025_meshmask_v3.1_binary.nc \
  --output sections_out --skip-invalid
```

This reuses `physical_stability.gloens_npy.load_gloens_mean`; when the domain grid lacks depth, that module uses its configured static meshmask for 1D depth coordinates. Precomputed `.npy` is **ensemble mean only**. `--skip-invalid` skips sections that fall outside the grid, exceed the native-grid distance tolerance, or have missing values. Without it, those sections raise an error.

## Local 3D NetCDF (test your own model data)

```bash
python gloens_sections.py --store /path/to/gloens.nc \
  --init-date 20260101 --sections def_section.yaml \
  --section Pacific_137E --variables thetao --output sections_out
```

## Test without real GLOENS data

```bash
python test_gloens_sections.py
```

Note: The original `def_section.yaml` repeats `Drake_Passage` for two regions (Atlantic and Antarctic). The included corrected `def_section.yaml` renames the first to `Drake_Passage_Atlantic`; **both transects are kept**. The original YAML is preserved as `def_section_original.yaml` and the reader deliberately rejects duplicate keys.

**Other caveats:** Sections are sampled on a great circle between YAML endpoints, whereas historical observing sections (e.g. OVIDE) may follow multi-waypoint ship tracks. A true observational track requires waypoint support. High-latitude/near-antipodal endpoints may need manually defined waypoints. Unlike the supplied `PlotSection`, this script does its own plotting (Matplotlib pcolormesh) to better preserve native-grid missing points and reproducible sampled indices; the supplied test/demo code informed the expected section outputs. No production CloudFerro dataset was accessible in this test environment.