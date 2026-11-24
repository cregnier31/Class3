"""Monthly native-grid section means, one 2x2 figure per lead and variable."""
import argparse
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from gloens_sections import read_sections, great_circle, to_xyz, safe_name
from physical_stability.forecasts import (SYSTEMS, ENDPOINT, forecast_store,
    open_cloudferro, detect_fields, prepare_ts)
from physical_stability.gloens_npy import load_gloens_mean, DEFAULT_GLOENS_GRID
from physical_stability.masks import MASK_PATHS, MASK_CANDIDATES

def routes(ds, sections, spacing, tolerance):
    lon,lat=ds.nav_lon.values,ds.nav_lat.values
    valid=np.isfinite(lon)&np.isfinite(lat)
    cells=np.argwhere(valid)
    tree=cKDTree(to_xyz(lon[valid],lat[valid]))
    result={}
    for name,c in sections.items():
        lo,la=great_circle(c['lonmin'],c['latmin'],c['lonmax'],c['latmax'],spacing)
        xyz=to_xyz(lo,la)
        chord,idx=tree.query(xyz)
        dist=np.r_[0,np.cumsum(12742*np.arcsin(np.clip(np.linalg.norm(np.diff(xyz,axis=0),axis=1)/2,0,1)))]
        y,x=cells[idx].T
        result[name]=(y,x,lo,la,dist,12742*np.arcsin(np.clip(chord/2,0,1))<=tolerance)
    return result

def extract(ds, route, mask, var):
    y,x,lo,la,d,near=route
    index={'y':xr.DataArray(y,dims='station'),'x':xr.DataArray(x,dims='station')}
    a=ds[var].isel(index).transpose('deptht','station').values.astype(float)
    wet=mask.isel({mask.dims[-2]:index['y'],mask.dims[-1]:index['x']}).values
    a=np.where((wet==1)&near&np.isfinite(a)&(abs(a)<1e30),a,np.nan)
    return xr.DataArray(a,dims=('depth','station'),coords={'depth':ds.deptht.values,
        'station':np.arange(len(y)),'longitude':('station',lo),'latitude':('station',la),
        'distance_km':('station',d)},attrs=dict(ds[var].attrs),name=var)

def process(args, dates, sections):
    # Accumulate only extracted sections, not global monthly fields.
    acc={}
    for system in SYSTEMS:
        maskpath=Path(args.mask_dir)/MASK_PATHS[system].name if args.mask_dir else MASK_PATHS[system]
        with xr.open_dataset(maskpath) as md:
            mask=md[next(n for n in MASK_CANDIDATES if n in md)].squeeze(drop=True)
            if mask.ndim not in (2,3):
                raise ValueError(f'{system}: invalid mask dimensions')
            cached=None
            grid=None
            def consume(raw,date,lead):
                nonlocal cached,grid
                fields=detect_fields(raw,system)
                ds,_=prepare_ts(raw,fields,system)
                ds=ds.isel(time=0,drop=True)
                coords=(ds.nav_lon.values,ds.nav_lat.values,ds.deptht.values)
                if cached is None:
                    grid=tuple(v.copy() for v in coords)
                    cached=routes(ds,sections,args.spacing_km,args.max_distance_km)
                elif any(not np.array_equal(a,b,equal_nan=True) for a,b in zip(grid,coords)):
                    raise ValueError(f'{system}: grid changed between runs')
                if mask.shape[-2:]!=(ds.sizes['y'],ds.sizes['x']) or (mask.ndim==3 and mask.shape[0]!=ds.sizes['deptht']):
                    raise ValueError(f'{system}: static mask shape mismatch')
                for name,route in cached.items():
                    for var in args.variables:
                        a=extract(ds,route,mask,var)
                        key=(system,lead,name,var)
                        if key not in acc:
                            acc[key]=[a,np.zeros(a.shape),np.zeros(a.shape,dtype=np.int16),[]]
                        _,total,count,used=acc[key]
                        good=np.isfinite(a.values)
                        total[good]+=a.values[good]
                        count[good]+=1
                        used.append(date)
                print(system,date,lead,flush=True)
            for date in dates:
                if system=='GLOENS':
                    for lead in args.leads:
                        raw=load_gloens_mean(args.npy_dir,args.grid_file,date,lead,include_currents=False)
                        consume(raw,date,lead)
                else:
                    with open_cloudferro(forecast_store(system,date),endpoint=args.endpoint,anonymous=not args.authenticated) as raw:
                        for lead in args.leads:
                            consume(raw.isel(time=slice(lead,lead+1)),date,lead)
    return acc

def geo(v,positive,negative):
    return f'{abs(v):.0f}°{positive if v>=0 else negative}'

def output(args,month,acc):
    means={}
    for key,(template,total,count,dates) in acc.items():
        a=template.copy(data=np.divide(total,count,out=np.full(total.shape,np.nan),where=count>=args.min_runs))
        a.attrs.update(initialization_month=month,initialization_dates=','.join(dates),run_count=len(dates))
        means[key]=a
        system,lead,name,var=key
        folder=Path(args.output)/month/safe_name(name)
        folder.mkdir(parents=True,exist_ok=True)
        xr.Dataset({var:a,'valid_run_count':(('depth','station'),count)}).to_netcdf(folder/f'{system}_{var}_lead{lead:03d}.nc')
    for name in sorted({k[2] for k in means}):
        for var in args.variables:
            all_arrays=[a.values for k,a in means.items() if k[2:]==(name,var)]
            finite=np.concatenate([v[np.isfinite(v)] for v in all_arrays])
            if not finite.size:
                print('No valid values:',name,var,flush=True)
                continue
            bounds=args.temperature_range if var=='thetao' else args.salinity_range
            low,high=bounds if bounds else (float(finite.min()),float(finite.max()))
            if low==high: high=low+0.01
            levels=np.linspace(low,high,21)
            for lead in args.leads:
                fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
                for ax,system in zip(axes.flat,SYSTEMS):
                    a=means[system,lead,name,var]
                    d=a.distance_km.values
                    z=a.depth.values
                    cs=ax.contourf(d,z,np.ma.masked_invalid(a.values),levels=levels,cmap='RdBu_r',extend='both',corner_mask=False)
                    ax.set_ylim(min(args.max_depth,float(z.max())),0)
                    ax.set_yscale('symlog',linthresh=20)
                    ticks=[v for v in (0,20,50,100,200,500,1000,2000,5000) if v<=min(args.max_depth,z.max())]
                    ax.set_yticks(ticks,labels=[str(v) for v in ticks])
                    idx=np.linspace(0,len(d)-1,6).astype(int)
                    ax.set_xticks(d[idx],labels=[geo(a.longitude.values[i],'E','W')+' / '+geo(a.latitude.values[i],'N','S') for i in idx],fontsize=8)
                    ax.set_xlabel('Longitude / latitude')
                    ax.set_ylabel('Depth (m)')
                    ax.set_title(f'{system} | {a.attrs["run_count"]} runs',fontweight='bold')
                    good=a.values[np.isfinite(a.values)]
                    if good.size:
                        ax.text(.5,-.20,f'Min {good.min():.2f}   Mean {good.mean():.2f}   Max {good.max():.2f}',transform=ax.transAxes,ha='center',fontsize=9)
                fig.colorbar(cs,ax=axes.ravel().tolist(),label='Temperature (°C)' if var=='thetao' else 'Salinity',shrink=.85)
                fig.suptitle(f'{name} | {var} | {month} | Lead {lead}',fontweight='bold',fontsize=16)
                fig.savefig(Path(args.output)/month/safe_name(name)/f'{var}_lead{lead:03d}_multimodel.png',dpi=160)
                plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npy-dir',default='/sortref/calval/CLASS4_scores/')
    p.add_argument('--grid-file',default=DEFAULT_GLOENS_GRID)
    p.add_argument('--mask-dir')
    p.add_argument('--sections',default='def_section.yaml')
    p.add_argument('--section',action='append')
    p.add_argument('--month',required=True,help='YYYYMM initialization month')
    p.add_argument('--dates',nargs='+',help='Explicit common initialization dates YYYYMMDD')
    p.add_argument('--leads',nargs='+',type=int,default=[1,3,5,7])
    p.add_argument('--variables',nargs='+',choices=['thetao','so'],default=['thetao','so'])
    p.add_argument('--min-runs',type=int,default=3)
    p.add_argument('--spacing-km',type=float,default=25)
    p.add_argument('--max-distance-km',type=float,default=150)
    p.add_argument('--max-depth',type=float,default=2000)
    p.add_argument('--temperature-range',nargs=2,type=float)
    p.add_argument('--salinity-range',nargs=2,type=float)
    p.add_argument('--endpoint',default=ENDPOINT)
    p.add_argument('--authenticated',action='store_true')
    p.add_argument('--output',default='monthly_results')
    args=p.parse_args()
    dates=sorted(set(args.dates or [m.group(1) for f in Path(args.npy_dir).glob(f'glo4-ens50_ensmean_thetao_leadtime_{args.leads[0]:03d}_{args.month}*.npy') if (m:=re.search(r'_(\d{8})\.npy$',f.name))]))
    if len(dates)<args.min_runs or any(not d.startswith(args.month) for d in dates):
        p.error('Insufficient initialization dates or dates outside month')
    sections=read_sections(args.sections)
    if args.section: sections={n:sections[n] for n in args.section}
    print('Common initialization dates:',dates,flush=True)
    output(args,args.month,process(args,dates,sections))

if __name__=='__main__':
    main()
