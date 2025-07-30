import numpy as np
from numpy import ma
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as polyg

from palettable.colorbrewer.diverging import RdBu_11, PuOr_11

import sys

font = {'family' : 'serif', 'size' : 9}
plt.rc('font', **font)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

#-----------------------------------------------------------------------------
# Define directories
# where data files ared
voldir = '/Volumes/SamT5/PhD/PhD_data/'
lmdir = voldir + 'land_masks/'
coastdir = voldir + 'land_masks/holland_vic/'
topodir = voldir + 'topog/'

contour = xr.open_dataset(topodir + 'bathy_mask_2km.nc')
cglat, cglon = np.meshgrid(contour.lat, contour.lon)
shelf_mask = contour.bathy_mask.values


# 2. MAP area and GRID
#------------------------------------------------------------------
# coastline resolution can vary from coarse to fine: c, l, i, h, f
plt.ioff()
m = Basemap(projection='spstere',
            boundinglat=-50,
            lon_0=180,
            resolution='i',
            round=True)

# extract segments north of 60 S
#print("Getting coastlines north of 60 S ... \n")
coast = m.drawcoastlines(linewidth=0)
segments = coast.get_segments()
lat_seg, lon_seg = [], []

for j in range(len(segments)):
    xy = np.vstack(segments[j]) 
    lons_b, lats_b = m(xy[:, 0], xy[:, 1], inverse=True)
    lats_bm = ma.masked_outside(lats_b, -60, -50.)
    lons_bm = ma.masked_array(lons_b, lats_bm.mask)
    if lats_bm.count() > 0:
        lat_seg.append(lats_bm)
        lon_seg.append(lons_bm)

# -----------------------
# coastlines south of 60S
#print("Importing Antarctic Digital Database shapefile. \n")
# add path to coastline files
sys.path.append(coastdir)
from coastline_Antarctica import coastline

# extract coastline points from files (Tiago/Paul Holland)
## it returns a list of lists
[ilon_land, ilat_land, ilon_ice, ilat_ice] = coastline()


def spstere_plot(varlon, varlat, var, vlims, cmap, cbar_units, bcolor):
    fig, ax = plt.subplots(figsize=(5, 5))
    cs = m.pcolormesh(varlon, varlat, var, 
                      vmin=vlims[0], vmax=vlims[1],
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True,
                      shading='auto')
    m.drawcoastlines(linewidth=0.25, zorder=3)
    m.fillcontinents(color='w')
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.09)
    cb.ax.set_title(cbar_units)
    
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=3, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightsteelblue',
                    zorder=3, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=3, c='dimgrey')

    m.plot(contour.llon, contour.llat,
            latlon=True, 
            color=bcolor, zorder=3, lw=1)
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=2, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=2, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, 1])
    return fig, ax, m



#------------------------------------------------------------------
def spstere_plot_old(varlon, varlat, var, vlims, cmap, cbar_units, cbar_extend):
    #plt.ion() (6.5, 7)
    fig, ax = plt.subplots(figsize=(4.5, 5), dpi=300)
    cs = m.pcolormesh(varlon, varlat, var,
                      vmin=vlims[0], vmax=vlims[1],
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True,
                      shading='auto')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='w')
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.07,
                      extend=cbar_extend)
    cb.ax.set_title(cbar_units)
    
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=1, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightgrey',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, c='dimgrey')
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=10, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=10, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)

    #m.drawmeridians([0, 160, -150, -62], linewidth=2, ax=ax)
    x1, y1 = m(190, -80.5)
    ax.annotate(r"$80^\circ$S", xy=(x1, y1),
                xycoords='data', xytext=(x1, y1),
                textcoords='data', zorder=10)
    x2, y2 = m(186, -70.5)
    ax.annotate(r"$70^\circ$S", xy=(x2, y2),
                xycoords='data', xytext=(x2, y2),
                textcoords='data', zorder=10)
    x3, y3 = m(184, -60.5)
    ax.annotate(r"$60^\circ$S", xy=(x3, y3),
                xycoords='data', xytext=(x3, y3),
                textcoords='data', zorder=10)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, 1])
    return fig, ax, m

#------------------------------------------------------------------
def spstere_contourf(varlon, varlat, var, contourf_kw, cmap, cbar_units):
    #plt.ion()
    fig, ax = plt.subplots(figsize=(6.5, 7))
    cs = m.contourf(varlon, varlat, var,
                      contourf_kw=contourf_kw,
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True, shading='auto')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='w')
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.07,
                      extend=cbar_extend)
    cb.ax.set_title(cbar_units)
    
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=1, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightgrey',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, c='dimgrey')
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=10, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=10, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)

    #m.drawmeridians([0, 160, -150, -62], linewidth=2, ax=ax)
    x1, y1 = m(190, -80.5)
    ax.annotate(r"$80^\circ$S", xy=(x1, y1),
                xycoords='data', xytext=(x1, y1),
                textcoords='data', zorder=10)
    x2, y2 = m(186, -70.5)
    ax.annotate(r"$70^\circ$S", xy=(x2, y2),
                xycoords='data', xytext=(x2, y2),
                textcoords='data', zorder=10)
    x3, y3 = m(184, -60.5)
    ax.annotate(r"$60^\circ$S", xy=(x3, y3),
                xycoords='data', xytext=(x3, y3),
                textcoords='data', zorder=10)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, 1])
    return fig, ax, m

def spstere_plot_nofig(fig, ax, varlon, varlat, var, vlims, cmap, cbar_units):
    m = Basemap(projection='spstere',
                boundinglat=-50,
                lon_0=180,
                resolution='i',
                round=True, ax=ax)
    cs = m.pcolormesh(varlon, varlat, var,
                      vmin=vlims[0], vmax=vlims[1],
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True, shading='auto')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='w')
    """
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.07,
                      extend='both')
    cb.ax.set_title(cbar_units)
    """
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=1, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightgrey',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, c='dimgrey')
    """ 
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=10, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=10, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)

    #m.drawmeridians([0, 160, -150, -62], linewidth=2, ax=ax)
    x1, y1 = m(190, -80.5)
    ax.annotate(r"$80^\circ$S", xy=(x1, y1),
                xycoords='data', xytext=(x1, y1),
                textcoords='data', zorder=10)
    x2, y2 = m(186, -70.5)
    ax.annotate(r"$70^\circ$S", xy=(x2, y2),
                xycoords='data', xytext=(x2, y2),
                textcoords='data', zorder=10)
    x3, y3 = m(184, -60.5)
    ax.annotate(r"$60^\circ$S", xy=(x3, y3),
                xycoords='data', xytext=(x3, y3),
    """
    ax.set_rasterization_zorder(0)

    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    plt.tight_layout()
    return fig, ax, m, cs
def spstere_frame():
    #plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightsteelblue',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, c='dimgrey')
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=1, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=1, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, .98])
    return fig, ax, m
