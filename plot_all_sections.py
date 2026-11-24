"""Global overview of all configured Class3 ocean sections."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from gloens_sections import read_sections, section_route

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sections',default='def_section.yaml')
    p.add_argument('--output',default='all_sections_map.png')
    args=p.parse_args()
    sections=read_sections(args.sections)
    colors={'All':'#555555','Arctic':'#8b36b5','Atlantic':'#0072b2','Pacific':'#d55e00','Indian':'#009e73','Mediterranean':'#cc3377','Antarctic':'#a6761d'}
    fig=plt.figure(figsize=(23,15))
    ax=fig.add_axes([.035,.25,.93,.68],projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN,facecolor='#eaf3fa')
    ax.add_feature(cfeature.LAND,facecolor='#d8d8d2')
    ax.coastlines(linewidth=.45)
    ax.gridlines(linewidth=.35,alpha=.4,linestyle=':')
    for i,(name,cfg) in enumerate(sections.items(),1):
        lo,la=section_route(cfg,25)
        color=colors.get(cfg.get('area'),'black')
        # Split at the dateline to avoid spurious lines across the map.
        breaks=np.flatnonzero(np.abs(np.diff(lo))>180)+1
        for x,y in zip(np.split(lo,breaks),np.split(la,breaks)):
            ax.plot(x,y,color=color,lw=1.7,transform=ccrs.PlateCarree(),zorder=4)
        j=len(lo)//2
        ax.text(lo[j],la[j],str(i),transform=ccrs.PlateCarree(),fontsize=8,color=color,weight='bold',ha='center',va='center',zorder=5,path_effects=[pe.withStroke(linewidth=2.5,foreground='white')])
    # Polar inset makes the explicit North Pole crossing easier to inspect.
    polar=fig.add_axes([.79,.73,.17,.19],projection=ccrs.NorthPolarStereo())
    polar.set_extent([-180,180,65,90],ccrs.PlateCarree())
    polar.add_feature(cfeature.LAND,facecolor='#d8d8d2')
    polar.coastlines(linewidth=.4)
    if 'Trans_arc' in sections:
        lo,la=section_route(sections['Trans_arc'],25)
        polar.plot(lo,la,color=colors['Arctic'],lw=2,transform=ccrs.Geodetic())
    polar.set_title('Trans_arc | North Pole',fontsize=11)
    for i,(name,cfg) in enumerate(sections.items()):
        col,row=divmod(i,13)
        fig.text(.035+col*.245,.225-row*.0145,f'{i+1:02d}  {name}',fontsize=10,color=colors.get(cfg.get('area'),'black'))
    fig.suptitle(f'Class3 | Global ocean sections ({len(sections)})',fontsize=25,weight='bold',y=.97)
    fig.legend(handles=[plt.Line2D([0],[0],color=c,lw=3,label=n) for n,c in colors.items()],loc='lower center',ncol=7,frameon=False,fontsize=12)
    fig.savefig(args.output,dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'Saved {args.output}')
if __name__=='__main__':
    main()
