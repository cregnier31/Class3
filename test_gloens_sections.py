"""Synthetic tests without production GLOENS data."""
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import Namespace
import numpy as np
import xarray as xr
from gloens_sections import great_circle,native_indices,section_extract,read_sections,plot_section

def test_geometry():
    lon,lat=great_circle(170,0,-170,0,50)
    assert len(lon)>2 and np.max(abs(lat))<1e-8
    assert np.max(abs(np.diff(lon)))>300
    assert len(great_circle(-180,0,180,0)[0])>1000
    assert len(great_circle(-155,-90,-155,90)[0])>500
    lon2,lat2=np.meshgrid(np.linspace(-10,10,61),np.linspace(-8,8,45))
    y,x,d,_=native_indices(lon2,lat2,np.ones(lon2.shape,bool),*great_circle(-9,-7,9,7,40),150)
    assert d.size==x.size and np.all(np.diff(d)>0)
    return lon2,lat2,y,x,d

def test_plot():
    lon,lat,y,x,d=test_geometry()
    z=xr.DataArray([0.,10.,50.,100.,300.],dims='depth')
    a=xr.DataArray(np.random.default_rng(42).normal(size=(5,*lon.shape)),dims=('depth','y','x'),name='thetao')
    sec=section_extract(a,z,xr.DataArray(lon,dims=('y','x')),xr.DataArray(lat,dims=('y','x')),y,x,d)
    assert sec.thetao.shape==(5,len(x))
    with TemporaryDirectory() as tmp:
        p=Path(tmp)
        sec.to_netcdf(p/'sec.nc')
        plot_section(sec,'thetao','Synthetic test',p/'sec.png',Namespace(cmap='viridis',vmin=None,vmax=None,max_depth=None,depth_scale='linear'))
        assert (p/'sec.png').stat().st_size>1000
    sections=read_sections(Path(__file__).with_name('def_section.yaml'))
    assert len(sections)==51
    assert 'Drake_Passage_Atlantic' in sections and 'Drake_Passage' in sections
    try:
        read_sections(Path(__file__).with_name('def_section_original.yaml'))
    except ValueError as exc:
        assert 'Duplicate YAML' in str(exc)
    else:
        raise AssertionError('Duplicate sections not rejected')

if __name__=='__main__':
    test_plot()
    print('Synthetic tests passed')
