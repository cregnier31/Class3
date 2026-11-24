"""Monthly native-grid section means, one 2x2 figure per lead and variable."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import ast
import configparser
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from gloens_sections import read_sections, section_route, to_xyz, safe_name
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
        lo,la=section_route(c,spacing)
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

def preflight(args, dates):
    """Pair weekly runs at equal forecast lead indices."""
    pairs=[]
    for date in dates:
        remote=(datetime.strptime(date,'%Y%m%d')-timedelta(days=1)).strftime('%Y%m%d')
        targets={lead:np.datetime64(datetime.strptime(date,'%Y%m%d'),'D')+np.timedelta64(lead,'D') for lead in args.leads}
        plan={}
        try:
            for lead in args.leads:
                with load_gloens_mean(args.npy_dir,args.grid_file,date,lead,include_currents=False) as raw:
                    stamp=np.datetime64(raw.time.values[0])
                    if stamp.astype('datetime64[D]')!=targets[lead]:
                        raise ValueError(f'GLOENS unexpected valid date {stamp}')
                    plan['GLOENS',lead]=(date,lead,str(stamp))
            for system in SYSTEMS:
                if system=='GLOENS': continue
                with open_cloudferro(forecast_store(system,remote),endpoint=args.endpoint,anonymous=not args.authenticated) as raw:
                    times=np.asarray(raw.time.values)
                    for lead in args.leads:
                        if not 0<=lead<len(times):
                            raise ValueError(f'{system}: unavailable lead {lead}')
                        target=np.datetime64(datetime.strptime(remote,'%Y%m%d'),'D')+np.timedelta64(lead,'D')
                        if np.datetime64(times[lead],'D')!=target:
                            raise ValueError(f'{system}: unexpected valid day at lead {lead}')
                        plan[system,lead]=(remote,lead,str(times[lead]))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f'SKIP weekly pair {date}: {exc}',flush=True)
            continue
        except Exception as exc:
            from zarr.errors import GroupNotFoundError
            if not isinstance(exc,GroupNotFoundError): raise
            print(f'SKIP weekly pair {date}: {exc}',flush=True)
            continue
        pairs.append(plan)
        print('Paired week',date,plan,flush=True)
    if len(pairs)<args.min_runs:
        raise ValueError(f'Only {len(pairs)} complete weekly pairs; need {args.min_runs}')
    return pairs

def process_models(args, pairs, sections, systems):
    # Accumulate only extracted sections, not global monthly fields.
    acc={}
    for system in systems:
        maskpath=Path(args.mask_dir)/MASK_PATHS[system].name if args.mask_dir else MASK_PATHS[system]
        with xr.open_dataset(maskpath) as md:
            mask=md[next(n for n in MASK_CANDIDATES if n in md)].squeeze(drop=True)
            if mask.ndim not in (2,3):
                raise ValueError(f'{system}: invalid mask dimensions')
            cached=None
            grid=None
            def consume(raw,date,lead,index,stamp):
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
                        used.append(dict(initialization_date=date,lead_index=index,valid_time=stamp))
                print(system,date,lead,flush=True)
            for plan in pairs:
                if system=='GLOENS':
                    for lead in args.leads:
                        date,index,stamp=plan[system,lead]
                        with load_gloens_mean(args.npy_dir,args.grid_file,date,index,include_currents=False) as raw:
                            consume(raw,date,lead,index,stamp)
                else:
                    date=plan[system,args.leads[0]][0]
                    with open_cloudferro(forecast_store(system,date),endpoint=args.endpoint,anonymous=not args.authenticated) as raw:
                        for lead in args.leads:
                            date,index,stamp=plan[system,lead]
                            consume(raw.isel(time=slice(index,index+1)),date,lead,index,stamp)
    return acc

def process(args,pairs,sections):
    if args.workers==1:
        return process_models(args,pairs,sections,SYSTEMS)
    acc={}
    with ProcessPoolExecutor(max_workers=min(args.workers,len(SYSTEMS)),mp_context=get_context('spawn')) as pool:
        jobs=[pool.submit(process_models,args,pairs,sections,(system,)) for system in SYSTEMS]
        for job in jobs:
            acc.update(job.result())
    return acc

def geo(v,positive,negative):
    return f'{abs(v):.0f}°{positive if v>=0 else negative}'

def add_locator(fig,lon,lat,name):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    xyz=to_xyz(lon,lat).mean(axis=0)
    if np.linalg.norm(xyz)<1e-6:
        center_lon,center_lat=float(lon[len(lon)//2]),float(lat[len(lat)//2])
    else:
        center_lon=np.degrees(np.arctan2(xyz[1],xyz[0]))
        center_lat=np.degrees(np.arctan2(xyz[2],np.hypot(xyz[0],xyz[1])))
    globe=fig.add_axes([.85,.14,.12,.16],projection=ccrs.Orthographic(center_lon,center_lat))
    globe.set_global()
    globe.add_feature(cfeature.OCEAN.with_scale('110m'),facecolor='#a6cee3')
    globe.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#e4dfc5')
    globe.coastlines(resolution='110m',linewidth=.35)
    globe.plot(lon,lat,color='red',linewidth=1.6,transform=ccrs.Geodetic())
    globe.set_title(name,fontsize=13)

def output(args,month,acc):
    means={}
    for key,(template,total,count,dates) in acc.items():
        a=template.copy(data=np.divide(total,count,out=np.full(total.shape,np.nan),where=count>=args.min_runs))
        a.attrs.update(initialization_month=month,initialization_dates=','.join(d['initialization_date'] for d in dates),run_count=len(dates), actual_lead_indices=','.join(str(d['lead_index']) for d in dates), valid_times=','.join(d['valid_time'] for d in dates), temporal_alignment='Same forecast lead index; valid dates may differ', reference_lead_index=key[1], run_metadata=json.dumps(dates))
        means[key]=a
        system,lead,name,var=key
        folder=Path(args.output)/month/safe_name(name)
        folder.mkdir(parents=True,exist_ok=True)
        xr.Dataset({var:a,'valid_run_count':(('depth','station'),count)}).to_netcdf(folder/f'{system}_{var}_lead{lead:03d}.nc')
    plot_means(args,month,means)

def plot_means(args,month,means):
    if args.plot_workers==1:
        return plot_sections(args,month,means)
    # Initialize Cartopy assets once before concurrent rendering.
    from cartopy.io import shapereader
    for feature in ('land','ocean','coastline'):
        shapereader.natural_earth(resolution='110m',category='physical',name=feature)
    names=sorted({k[2] for k in means})
    with ProcessPoolExecutor(max_workers=min(args.plot_workers,len(names)),mp_context=get_context('spawn')) as pool:
        jobs=[pool.submit(plot_sections,args,month,{k:a for k,a in means.items() if k[2]==name}) for name in names]
        for job in jobs:
            job.result()

def plot_sections(args,month,means):
    section_configs=read_sections(args.sections)
    cfg=configparser.ConfigParser(interpolation=None)
    if not cfg.read(args.params):
        raise FileNotFoundError(args.params)
    cmin=ast.literal_eval(cfg['dict']['cmin'])
    cmax=ast.literal_eval(cfg['dict']['cmax'])
    def get_palette(var):
        cmap_name=args.cmap or ('thermal' if var=='thetao' else 'haline')
        if cmap_name=='IDL_micom34':
            rgb=np.loadtxt(Path(__file__).with_name('IDL_micom34.pal'))
            if rgb.ndim!=2 or rgb.shape[1]!=3 or np.any((rgb<0)|(rgb>255)):
                raise ValueError('Invalid IDL RGB palette')
            cmap=ListedColormap(rgb/255.,name=cmap_name)
        elif cmap_name in ('thermal','haline'):
            import cmocean
            cmap=getattr(cmocean.cm,cmap_name)
        else:
            cmap=plt.get_cmap(cmap_name)
        return cmap
    for name in sorted({k[2] for k in means}):
        max_depth=min(args.max_depth,float(section_configs[name].get('max_depth',args.max_depth)))
        for var in args.variables:
            cmap=get_palette(var)
            all_arrays=[a.values for k,a in means.items() if k[2:]==(name,var)]
            finite=np.concatenate([v[np.isfinite(v)] for v in all_arrays])
            if not finite.size:
                print('No valid values:',name,var,flush=True)
                continue
            bounds=args.temperature_range if var=='thetao' else args.salinity_range
            field='temperature' if var=='thetao' else 'salinity'
            step=args.temperature_step if var=='thetao' else args.salinity_step
            if step<=0: raise ValueError('Contour step must be positive')
            if bounds:
                low,high=bounds
            elif field in cmin.get(name,{}) and field in cmax.get(name,{}):
                low,high=cmin[name][field],cmax[name][field]
            else:
                low=np.floor(finite.min()/step)*step
                high=np.ceil(finite.max()/step)*step
                print(f'WARNING: no configured limits for {name} {field}; using data range',flush=True)
                if low==high: high=low+step
            if high<=low: raise ValueError(f'Invalid color limits for {name} {field}')
            interior=np.arange(np.floor(low/step)+1,np.ceil(high/step))*step
            levels=np.r_[low,interior[(interior>low+1e-9)&(interior<high-1e-9)],high]
            tickstep=1. if var=='thetao' else .5
            cb_ticks=np.arange(np.ceil(low/tickstep),np.floor(high/tickstep)+1)*tickstep
            print(f'Color limits {name} {field}: {low:g}, {high:g}',flush=True)
            for lead in args.leads:
                fig,axes=plt.subplots(2,2,figsize=(19,14))
                fig.subplots_adjust(left=.07,right=.81,bottom=.14,top=.90,wspace=.28,hspace=.65)
                for ax,system in zip(axes.flat,SYSTEMS):
                    a=means[system,lead,name,var]
                    d=a.distance_km.values
                    z=a.depth.values
                    cs=ax.contourf(d,z,np.ma.masked_invalid(a.values),levels=levels,cmap=cmap,extend='both',corner_mask=False)
                    if not args.no_contours:
                        line_step=args.temperature_contour_step if var=='thetao' else args.salinity_contour_step
                        if line_step<=0:
                            raise ValueError('Contour line interval must be positive')
                        values=a.values[np.isfinite(a.values)]
                        if values.size:
                            line_levels=np.arange(np.ceil(low/line_step),np.floor(high/line_step)+1)*line_step
                            line_levels=line_levels[(line_levels>values.min())&(line_levels<values.max())]
                            if line_levels.size:
                                line_colors=args.contour_color
                                if args.contour_color in ('colorbar','colorbar-dark'):
                                    line_colors=cs.cmap(cs.norm(line_levels))
                                    if args.contour_color=='colorbar-dark':
                                        line_colors[:,:3]*=1-np.clip(args.contour_darkening,0,1)
                                ax.contour(d,z,np.ma.masked_invalid(a.values),levels=line_levels,
                                           colors=line_colors,linewidths=args.contour_linewidth+(args.contour_max_linewidth-args.contour_linewidth)*np.clip((line_levels-low)/max(high-low,1e-12),0,1),
                                           linestyles=args.contour_linestyle,alpha=1.,corner_mask=False)
                    ax.set_ylim(max_depth,args.min_depth)
                    ax.set_yscale('symlog',linthresh=20)
                    ticks=[v for v in (0,20,50,100,200,500,1000,2000,5000) if args.min_depth<=v<=max_depth]
                    ax.set_yticks(ticks,labels=[str(v) for v in ticks])
                    idx=np.linspace(0,len(d)-1,6).astype(int)
                    ax.set_xticks(d[idx],labels=[geo(a.longitude.values[i],'E','W') for i in idx],fontsize=13)
                    ax.set_xlabel('Longitude',fontsize=15,fontweight='bold')
                    sec=ax.secondary_xaxis('bottom')
                    sec.spines['bottom'].set_position(('outward',42))
                    sec.set_xticks(d[idx],labels=[geo(a.latitude.values[i],'N','S') for i in idx],fontsize=13)
                    sec.set_xlabel('Latitude',fontsize=15,fontweight='bold')
                    ax.set_ylabel('Depth (m)',fontsize=15,fontweight='bold')
                    ax.tick_params(axis='both',labelsize=14)
                    sec.tick_params(axis='x',labelsize=14)
                    ax.set_title(f'{system} | {a.attrs["run_count"]} runs | leadtime {lead} days',fontweight='bold',fontsize=17)
                    good=a.values[np.isfinite(a.values)]
                    if good.size:
                        ax.text(.5,-.36,f'Min {good.min():.2f}   Mean {good.mean():.2f}   Max {good.max():.2f}',transform=ax.transAxes,ha='center',fontsize=13)
                cax=fig.add_axes([.845,.34,.022,.55])
                cb=fig.colorbar(cs,cax=cax,label='Temperature (°C)' if var=='thetao' else 'Salinity (PSU)',ticks=cb_ticks,spacing='proportional',format=mticker.FormatStrFormatter('%g'))
                cb.ax.tick_params(labelsize=14)
                cb.ax.yaxis.label.set_size(16)
                cb.ax.yaxis.label.set_weight('bold')
                add_locator(fig,a.longitude.values,a.latitude.values,name)
                fig.suptitle(f'{name} | {var} | {month} | Forecast lead D+{lead} | same leadtime',fontweight='bold',fontsize=22)
                fig.savefig(Path(args.output)/month/safe_name(name)/f'{var}_lead{lead:03d}_multimodel.png',dpi=200)
                plt.close(fig)

def load_monthly_means(args,sections):
    means={}
    for name in sections:
        folder=Path(args.output)/args.month/safe_name(name)
        for system in SYSTEMS:
            for lead in args.leads:
                for var in args.variables:
                    path=folder/f'{system}_{var}_lead{lead:03d}.nc'
                    if not path.is_file():
                        raise FileNotFoundError(f'Missing monthly section: {path}')
                    with xr.open_dataset(path) as ds:
                        a=ds[var].load()
                    if a.attrs.get('temporal_alignment')!='Same forecast lead index; valid dates may differ' or any(int(i)!=lead for i in a.attrs.get('actual_lead_indices','-1').split(',')):
                        raise ValueError(f'{path}: not computed at equal forecast leads; recompute monthly sections')
                    means[system,lead,name,var]=a
    return means

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npy-dir',default='/sortref/calval/CLASS4_scores/')
    p.add_argument('--grid-file',default=DEFAULT_GLOENS_GRID)
    p.add_argument('--mask-dir')
    p.add_argument('--sections',default='def_section.yaml')
    p.add_argument('--cmap',default='jet',help='Colormap (default jet; override with another Matplotlib colormap)')
    p.add_argument('--params',default=str(Path(__file__).with_name('params_section.cfg')))
    p.add_argument('--section',action='append')
    p.add_argument('--month',required=True,help='YYYYMM initialization month')
    p.add_argument('--dates',nargs='+',help='Explicit GLOENS initialization dates YYYYMMDD')
    p.add_argument('--leads',nargs='+',type=int,default=[1,3,5,7])
    p.add_argument('--variables',nargs='+',choices=['thetao','so'],default=['thetao','so'])
    p.add_argument('--min-runs',type=int,default=3)
    p.add_argument('--spacing-km',type=float,default=25)
    p.add_argument('--max-distance-km',type=float,default=150)
    p.add_argument('--max-depth',type=float,default=2000)
    p.add_argument('--min-depth',type=float,default=0)
    p.add_argument('--temperature-step',type=float,default=.25)
    p.add_argument('--salinity-step',type=float,default=.125)
    p.add_argument('--temperature-range',nargs=2,type=float)
    p.add_argument('--salinity-range',nargs=2,type=float)
    p.add_argument('--endpoint',default=ENDPOINT)
    p.add_argument('--authenticated',action='store_true')
    p.add_argument('--no-contours',action='store_true',help='Disable contour lines')
    p.add_argument('--temperature-contour-step',type=float,default=2.)
    p.add_argument('--salinity-contour-step',type=float,default=.25)
    p.add_argument('--contour-color',default='colorbar-dark',help='Contour line color: colorbar-dark, colorbar, white, black or another Matplotlib color')
    p.add_argument('--contour-linestyle',default='dashed',choices=['solid','dashed','dotted','dashdot'])
    p.add_argument('--contour-darkening',type=float,default=.4,choices=None,help='Fraction of RGB intensity removed in colorbar-dark mode (0 to 1)')
    p.add_argument('--contour-max-linewidth',type=float,default=3.0)
    p.add_argument('--contour-linewidth',type=float,default=.5)
    p.add_argument('--plot-only',action='store_true',help='Regenerate PNGs from existing monthly NetCDF files')
    p.add_argument('--output',default='monthly_results_same_lead')
    p.add_argument('--workers',type=int,default=4,help='Parallel model processes (1 for sequential)')
    p.add_argument('--plot-workers',type=int,default=8,help='Parallel section plotting processes (1 for sequential)')
    args=p.parse_args()
    if args.workers<1 or args.plot_workers<1:
        p.error('Worker counts must be positive')
    if args.plot_only:
        sections=read_sections(args.sections)
        if args.section: sections={n:sections[n] for n in args.section}
        plot_means(args,args.month,load_monthly_means(args,sections))
        return
    dates=sorted(set(args.dates or [m.group(1) for f in Path(args.npy_dir).glob(f'glo4-ens50_ensmean_thetao_leadtime_{args.leads[0]:03d}_{args.month}*.npy') if (m:=re.search(r'_(\d{8})\.npy$',f.name))]))
    if len(dates)<args.min_runs or any(not d.startswith(args.month) for d in dates):
        p.error('Insufficient initialization dates or dates outside month')
    sections=read_sections(args.sections)
    if args.section: sections={n:sections[n] for n in args.section}
    print('Common initialization dates:',dates,flush=True)
    output(args,args.month,process(args,preflight(args,dates),sections))

if __name__=='__main__':
    main()
