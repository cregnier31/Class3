#!/usr/bin/env python3
"""Native T-grid GLOENS tracer sections (not transport sections)."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import re
import sys

# Reuse the neighbouring physical_stability checkout without installation.
_shared_src = Path(__file__).resolve().parent.parent / "physical_stability" / "src"
if (_shared_src / "physical_stability").is_dir():
    sys.path.insert(0, str(_shared_src))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from scipy.spatial import cKDTree

def read_sections(path):
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader,node):
        result={}
        for k,v in node.value:
            key=loader.construct_object(k)
            if key in result:
                raise ValueError(f'Duplicate YAML section name {key!r}')
            result[key]=loader.construct_object(v)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    with open(path,encoding='utf8') as f:
        sections=yaml.load(f,Loader=UniqueLoader)
    if not isinstance(sections,dict) or not sections:
        raise ValueError('Expected nonempty section mapping')
    for name,cfg in sections.items():
        if 'lonvalues' in cfg or 'latvalues' in cfg:
            lo=np.asarray(cfg.get('lonvalues',[]),dtype=float)
            la=np.asarray(cfg.get('latvalues',[]),dtype=float)
            if lo.ndim!=1 or lo.shape!=la.shape or lo.size<2 or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(la)) or np.any(np.abs(la)>90):
                raise ValueError(f'{name}: invalid explicit route')
            continue
        for k in ('lonmin','lonmax','latmin','latmax'):
            if k not in cfg or not np.isfinite(float(cfg[k])):
                raise ValueError(f'{name}: invalid {k}')
    return sections

def section_route(cfg, spacing):
    if 'lonvalues' in cfg:
        return np.asarray(cfg['lonvalues'],dtype=float),np.asarray(cfg['latvalues'],dtype=float)
    return great_circle(cfg['lonmin'],cfg['latmin'],cfg['lonmax'],cfg['latmax'],spacing)


def to_xyz(lon,lat):
    lon,lat=np.deg2rad(np.broadcast_arrays(lon,lat))
    return np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
        np.cos(lat).ravel()*np.sin(lon).ravel(),np.sin(lat).ravel()))

def great_circle(lon0,lat0,lon1,lat1,spacing_km=25.):
    if spacing_km<=0:
        raise ValueError('Positive spacing required')
    # Preserve zonal and meridional definitions, including full equator.
    if lat0==lat1:
        delta=lon1-lon0
        if abs(delta)<360:
            delta=(delta+180)%360-180
        n=max(2,int(np.ceil(abs(np.deg2rad(delta))*6371*np.cos(np.deg2rad(lat0))/spacing_km))+1)
        return (np.linspace(lon0,lon0+delta,n)+180)%360-180,np.full(n,lat0)
    if (lon1-lon0)%360==0:
        n=max(2,int(np.ceil(abs(np.deg2rad(lat1-lat0))*6371/spacing_km))+1)
        return np.full(n,lon0),np.linspace(lat0,lat1,n)
    a,b=to_xyz([lon0,lon1],[lat0,lat1])
    angle=float(np.arccos(np.clip(a@b,-1,1)))
    f=np.linspace(0,1,max(2,int(np.ceil(6371*angle/spacing_km))+1))
    if angle<1e-12:
        xyz=np.broadcast_to(a,(len(f),3))
    elif abs(np.pi-angle)<1e-7:
        raise ValueError('Antipodal endpoints require an intermediate waypoint')
    else:
        xyz=(np.sin((1-f)*angle)[:,None]*a+np.sin(f*angle)[:,None]*b)/np.sin(angle)
    return np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0])),np.rad2deg(np.arctan2(xyz[:,2],np.hypot(xyz[:,0],xyz[:,1])))

def native_indices(lon2d,lat2d,wet2d,slon,slat,max_distance_km):
    valid=np.isfinite(lon2d)&np.isfinite(lat2d)
    if not valid.any():
        raise ValueError('No valid grid coordinates')
    cells=np.argwhere(valid)
    chord,nearest=cKDTree(to_xyz(lon2d[valid],lat2d[valid])).query(to_xyz(slon,slat))
    error=6371*2*np.arcsin(np.clip(chord/2,0,1))
    if np.any(error>max_distance_km):
        raise ValueError(f'Route exceeds native-grid tolerance: {error.max():.1f} km')
    yx=cells[nearest]
    yx=yx[np.r_[True,np.any(yx[1:]!=yx[:-1],axis=1)]]
    if len(yx)<2:
        raise ValueError('Fewer than two distinct grid cells')
    y,x=yx.T
    xyz=to_xyz(lon2d[y,x],lat2d[y,x])
    d=6371*2*np.arcsin(np.clip(np.linalg.norm(np.diff(xyz,axis=0),axis=1)/2,0,1))
    return y,x,np.r_[0,np.cumsum(d)],float(error.max())

def detect_grid(ds,args):
    def find(override,names):
        name=override or next((n for n in names if n in ds),None)
        if name is None:
            raise KeyError('Missing grid coordinates')
        return ds[name].squeeze(drop=True)
    lon=find(args.longitude,('longitude','nav_lon','glamt','lon'))
    lat=find(args.latitude,('latitude','nav_lat','gphit','lat'))
    if lon.ndim!=2 or lon.dims!=lat.dims:
        raise ValueError('Expected matching 2D T grid')
    return lon,lat

def get_depth(da,ds,depth_name=None):
    for name in ((depth_name,) if depth_name else ('depth','deptht','deptho','z','lev','nav_lev')):
        if name in ds and ds[name].ndim==1 and ds[name].dims[0] in da.dims:
            return ds[name]
    raise ValueError('Missing fixed 1D depth coordinate')

def section_extract(da,depth,lon,lat,y,x,distance):
    yd,xd=lon.dims
    st=da.isel({yd:xr.DataArray(y,dims='station'),xd:xr.DataArray(x,dims='station')})
    st=st.where(np.isfinite(st)&(abs(st)<1e30))
    if 'ens' in st.dims:
        st=st.mean('ens',skipna=True)
    st=st.transpose(depth.dims[0],'station')
    result=xr.Dataset({str(da.name):(('depth','station'),np.asarray(st.values,dtype=float))},
        coords={'depth':np.asarray(depth.values,dtype=float),'station':np.arange(len(y)),
        'distance_km':('station',distance),'longitude':('station',lon.values[y,x]),
        'latitude':('station',lat.values[y,x]),'grid_y':('station',y.astype('int32')),
        'grid_x':('station',x.astype('int32'))})
    result[str(da.name)].attrs=dict(da.attrs)
    return result

def plot_section(section,variable,title,out,args):
    values=np.ma.masked_invalid(section[variable].values)
    if not values.count():
        raise ValueError('All extracted values missing')
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    try:
        cs=ax.pcolormesh(section.distance_km.values,section.depth.values,values,
            cmap=args.cmap,shading='nearest',vmin=args.vmin,vmax=args.vmax)
        ax.invert_yaxis()
        if args.depth_scale=='symlog':
            ax.set_yscale('symlog',linthresh=10)
        if args.max_depth is not None:
            ax.set_ylim(args.max_depth,0)
        ax.set(xlabel='Along-section distance (km)',ylabel='Depth (m)',title=title)
        fig.colorbar(cs,ax=ax,label=f"{variable} ({section[variable].attrs.get('units','')})")
        fig.savefig(out,dpi=160)
    finally:
        plt.close(fig)

@contextmanager
def open_source(args):
    if args.npy_dir:
        yield None
    elif args.store.endswith(('.nc','.nc4')):
        with xr.open_dataset(args.store,chunks='auto') as ds:
            yield ds
    elif Path(args.store).exists():
        with xr.open_zarr(args.store,consolidated=False,chunks='auto') as ds:
            yield ds
    else:
        from physical_stability.forecasts import open_cloudferro
        with open_cloudferro(args.store,endpoint=args.endpoint,anonymous=not args.authenticated) as ds:
            yield ds

def load_mask(args,shape):
    if not args.mask_file:
        return None
    with xr.open_dataset(args.mask_file) as ds:
        name=args.mask_variable or next((n for n in ('tmask','mask','mask_thetao','wet_mask','wet','tmaskutil') if n in ds),None)
        if name is None:
            raise KeyError('Missing mask variable')
        mask=np.asarray(ds[name].squeeze(drop=True).values)
    if mask.ndim not in (2,3) or mask.shape[-2:]!=shape or not np.all((mask==0)|(mask==1)):
        raise ValueError('Invalid binary T mask')
    return mask.astype(bool)

def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',name).strip('_')

def run(args):
    sections=read_sections(args.sections)
    if args.section:
        sections={n:sections[n] for n in args.section}
    out=Path(args.output)
    with open_source(args) as ds:
        nlead=28 if ds is None else ds.sizes['time']
        routes=None
        for lead in range(args.first_day,min(args.last_day,nlead)):
            if ds is None:
                from physical_stability.gloens_npy import load_gloens_mean
                day=load_gloens_mean(args.npy_dir,args.grid_file,args.init_date,lead,axis_order=args.npy_axes,include_currents=False)
                if 'time' in day.dims:
                    day=day.isel(time=0)
            else:
                day=ds.isel(time=lead)
            lon,lat=detect_grid(day,args)
            if routes is None:
                mask=load_mask(args,lon.shape)
                wet=np.ones(lon.shape,bool) if mask is None else (mask[0] if mask.ndim==3 else mask)
                routes={}
                for name,cfg in sections.items():
                    try:
                        route=section_route(cfg,args.spacing_km)
                        routes[name]=native_indices(lon.values,lat.values,wet,*route,args.max_distance_km)
                    except ValueError as exc:
                        if not args.skip_invalid:
                            raise
                        print(f'SKIPPED {name}: {exc}',file=sys.stderr)
            for name,(y,x,d,maxdist) in routes.items():
                for var in args.variables:
                    try:
                        da=day[var]
                        depth=get_depth(da,day,args.depth)
                        if 'ens' in da.dims:
                            if da.sizes['ens']!=50:
                                raise ValueError('Expected 50 ensemble members')
                            if args.ensemble=='members':
                                da=da.isel(ens=args.member)
                        elif args.ensemble=='members':
                            raise ValueError('Input lacks ensemble dimension')
                        sec=section_extract(da,depth,lon,lat,y,x,d)
                        if mask is not None and mask.ndim==2:
                            sec[var]=sec[var].where(xr.DataArray(mask[y,x],dims=('station',)))
                        if mask is not None and mask.ndim==3:
                            if mask.shape[0]!=sec.sizes['depth']:
                                raise ValueError('Mask depth mismatch')
                            sec[var]=sec[var].where(xr.DataArray(mask[:,y,x],dims=('depth','station')))
                        sec.attrs.update(source_system='GLOENS',source_store=str(args.store or args.npy_dir),
                            lead_index=lead,section_name=name,area=sections[name].get('area',''),
                            ensemble_processing=args.ensemble,maximum_route_to_grid_km=maxdist)
                        tag=f'{safe_name(name)}_{safe_name(var)}_lead{lead:03d}'
                        if args.ensemble=='members':
                            tag+=f'_member{args.member:02d}'
                        folder=out/safe_name(name)
                        folder.mkdir(parents=True,exist_ok=True)
                        sec.to_netcdf(folder/(tag+'.nc'))
                        plot_section(sec,var,f'GLOENS | {name} | {var} | lead {lead:03d}',folder/(tag+'.png'),args)
                    except (ValueError,KeyError) as exc:
                        if not args.skip_invalid:
                            raise
                        print(f'SKIPPED {name} {var} lead {lead}: {exc}',file=sys.stderr)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--store')
    source.add_argument('--npy-dir')
    p.add_argument('--grid-file')
    p.add_argument('--init-date',required=True)
    p.add_argument('--sections',default='def_section.yaml')
    p.add_argument('--section',action='append')
    p.add_argument('--variables',nargs='+',default=['thetao','so'])
    p.add_argument('--first-day',type=int,default=0)
    p.add_argument('--last-day',type=int,default=1)
    p.add_argument('--ensemble',choices=('mean','members'),default='mean')
    p.add_argument('--member',type=int)
    p.add_argument('--npy-axes',default='depth,y,x')
    for name in ('depth','longitude','latitude','mask-file','mask-variable'):
        p.add_argument('--'+name)
    p.add_argument('--spacing-km',type=float,default=25.)
    p.add_argument('--max-distance-km',type=float,default=150.)
    p.add_argument('--output',default='gloens_sections_output')
    p.add_argument('--cmap',default='viridis')
    p.add_argument('--depth-scale',choices=('linear','symlog'),default='linear')
    for name in ('max-depth','vmin','vmax'):
        p.add_argument('--'+name,type=float)
    p.add_argument('--skip-invalid',action='store_true')
    p.add_argument('--endpoint',default='https://s3.waw3-1.cloudferro.com')
    p.add_argument('--authenticated',action='store_true')
    args=p.parse_args(argv)
    datetime.strptime(args.init_date,'%Y%m%d')
    if args.npy_dir and (not args.grid_file or args.ensemble!='mean'):
        p.error('NPY requires grid file and mean mode')
    if args.ensemble=='members' and (args.member is None or not 0<=args.member<50):
        p.error('Specify member index 0..49')
    if not 0<=args.first_day<args.last_day or min(args.spacing_km,args.max_distance_km)<=0:
        p.error('Invalid lead window or sampling distances')
    run(args)

if __name__=='__main__':
    main()
