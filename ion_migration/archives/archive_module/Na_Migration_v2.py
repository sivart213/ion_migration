# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:42:38 2021

@author: j2cle
"""
#%%
import numpy as np
import pandas as pd
import General_functions as gf
import matplotlib.pyplot as plt
# import inspect
from matplotlib import ticker
from scipy.special import erfc
from scipy import optimize
# from scipy.io import savemat
# import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
# from matplotlib.ticker import ScalarFormatter
# import matplotlib as mpl
from scipy.optimize import curve_fit,fsolve,minimize
from functools import partial
from dataclasses import dataclass, field
from dataclasses import InitVar

def arrh(T,pre_fac,E_A):
    return pre_fac*np.exp(E_A/(gf.KB_EV*T))

def c_ratio_np(depth,thick,temp,e_app,time,diff,mob):
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (1/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))

def c_ratio_np_df(time,df):#thick,temp,e_app,time,diff,mob):
    mob=df['mob'].to_numpy()
    e_app=df['efield'].to_numpy()
    depth=df['depth'].to_numpy()
    diff=df['mob'].to_numpy()
    thick=df['thick'].to_numpy()
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (1/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))

def c_ratio_np_s(time,df):#thick,temp,e_app,time,diff,mob):
    mob=df['mob']
    e_app=df['efield']
    depth=df['depth']
    diff=df['mob']
    thick=df['thick']
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (1/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))

def find_tauc_np_cratio_3d(time,depth,temp,e_app,d_0,e_a, target=0.08, thick = 450e-4):
    diff = d_0*np.exp(-e_a/(gf.KB_EV*temp))
    mob = diff/(gf.KB_EV*temp)
    if depth == 0:
        depth = (2*np.sqrt(diff * time)) + mob * e_app * time
    if thick == 0:
        thick = (2*np.sqrt(diff * time)) + mob * e_app * time

    ratio = c_ratio_np(depth,thick,temp,e_app,time,diff,mob)
    if isinstance(ratio,float):
        if ratio <  1e-10:
            ratio = 1
    else:
        for n in range(len(ratio)):
            if ratio[n] <  1e-10:
                ratio[n] = 1

    return (ratio-target)**2
def find_tauc_np_cratio_3d_df(time, df, target=0.08):

    ratio = c_ratio_np_s(time, df)
    if isinstance(ratio,float):
        if ratio <  1e-10:
            ratio = 1
    else:
        for n in range(len(ratio)):
            if ratio[n] <  1e-10:
                ratio[n] = 1

    return (ratio-target)**2

def find_tauc_np_cratio_3d_alt(time,depth,temp,e_app,d_0,e_a, target=0.08, thick = 450e-4):
    diff = d_0*np.exp(-e_a/(gf.KB_EV*temp))
    mob = diff/(gf.KB_EV*temp)
    if depth == 0:
        depth = (2*np.sqrt(diff * time)) + mob * e_app * time
    if thick == 0:
        thick = (2*np.sqrt(diff * time)) + mob * e_app * time

    ratio = c_ratio_np(depth,thick,temp,e_app,time,diff,mob)

    return ratio
    # return abs(c_ratio_np(depth,thick,temp,e_app,time,diff,mob)-target)



def limitcontour(x,y,z, xlim=None, ylim=None, **kwargs):

    if len(x.shape)==1:
        xx,yy = np.meshgrid(x,y)
    else:
        xx,yy = x,y

    xmask = np.ones(x.shape).astype(bool)
    ymask = np.ones(y.shape).astype(bool)
    zmask = np.ones(z.shape).astype(bool)

    if xlim:
        xmask = xmask & (x>=xlim[0]) & (x<=xlim[1])
        zmask = zmask & (xx>=xlim[0]) & (xx<=xlim[1])

    if ylim:
        ymask = ymask & (y>=ylim[0]) & (y<=ylim[1])
        zmask = zmask & (yy>=ylim[0]) & (yy<=ylim[1])

    xm = np.ma.masked_where(~xmask , x)
    ym = np.ma.masked_where(~ymask , y)
    zm = np.ma.masked_where(~zmask , z)

    return xm,ym,zm


def plotcont6(x,y,z,rho1,rho2,v1,v2,v3,name='',x_corr=0,y_corr=1,xname='Temp',yname='E',ind=0,levels=50,templim=[25,85]):

    z=z/(3600*24)

    if x_corr == 273.15:
        xunit = ' [C]'
    else:
        xunit = ' [K]'
    if y_corr == 1e6:
        yunit = ' [MV/cm]'
    else:
        yunit = ' [V/cm]'

    if yname == 'V app':
        yunit = ' [V]'



    fig, ax = plt.subplots()
    csa = ax.contourf(x-x_corr,y/y_corr,z,np.logspace(np.log10(z.min()),np.log10(z.max()), levels), locator = ticker.LogLocator(),  cmap='gist_heat')

    ax.set_xlabel(xname+xunit)
    ax.set_xlim(templim[0],templim[1])
    ax.set_ylabel(yname+yunit)
    ax.set_yscale('log')
    ax.set_title(name)

    cbar = fig.colorbar(csa)
    cbar.locator = ticker.LogLocator(10)
    cbar.set_ticks(cbar.locator.tick_values(z.min(), z.max()))
    cbar.minorticks_off()
    cbar.set_label('Breakthrough [Days]')


    xx,yy,zz = limitcontour(x-x_corr,y/y_corr,z,xlim=templim)

    visual_levels = [1, 4, 7, 30, 365, 365*10]
    lv_lbls = ['1 d', '4 d', '1 w', '1 mo', '1 yr', '10 yr']
    ax = plt.gca()
    csb = ax.contour(xx,yy,zz,visual_levels, colors='w',locator=ticker.LogLocator(),linestyles='--',norm=LogNorm(),linewidths=1.25)
    csb.levels = lv_lbls

    ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=False)

    b1 = rho1*v1/y_corr
    b2 = rho1*v2/y_corr
    b3 = rho1*v3/y_corr

    points = [(30,b1[gf.find_nearest(x-x_corr, 30)]),(45,b2[gf.find_nearest(x-x_corr, 45)]),(60,b3[gf.find_nearest(x-x_corr, 60)])]
    v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # ax = plt.gca()
    ax.plot(x-x_corr,b1,x-x_corr,b2,x-x_corr,b3, color='k',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],textcoords='offset points',xytext=(0,40),ha='center',arrowprops=dict(facecolor='black', shrink=0.001),
    #         horizontalalignment='right', verticalalignment='bottom')
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="gray"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
    # c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr

    # points = [(35,c1[gf.find_nearest(x-x_corr, 35)]),(50,c2[gf.find_nearest(x-x_corr, 50)]),(65,c3[gf.find_nearest(x-x_corr, 65)])]
    # v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # # ax = plt.gca()
    # ax.plot(x-x_corr,c1,x-x_corr,c2,x-x_corr,c3, color='b',linestyle='--',linewidth=1.25)
    # for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
    #         bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
    #         xytext=(0,-25), textcoords='offset points', ha='center',
    #         arrowprops=dict(arrowstyle="->"))
    c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr

    points = [(35,c1[gf.find_nearest(x-x_corr, 35)])]
    v_lbls = [str(v1)+' V (Boro)']
    # ax = plt.gca()
    ax.plot(x-x_corr,c1, color='b',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
    return

def plotcont8(x,y,z,rho1,rho2,v1,v2,v3,name='',x_corr=0,y_corr=1,xname='Temperature',yname='E',ind=0,levels=50,templim=[25,85]):

    z=z/(3600*24)

    if x_corr == 273.15:
        xunit = ' ($^o$C)'
    else:
        xunit = ' (K)'
    if y_corr == 1e6:
        yunit = ' (MV/cm)'
    else:
        yunit = ' (V/cm)'

    if yname == 'V app':
        yunit = ' (V)'



    fig, ax = plt.subplots()
    csa = ax.contourf(x-x_corr,y/y_corr,z,np.logspace(np.log10(z.min()),np.log10(z.max()), levels), locator = ticker.LogLocator(),  cmap='gist_heat')

    ax.set_xlabel(xname+xunit, fontname='Arial', fontsize=18, fontweight='bold')
    ax.set_xlim(templim[0],templim[1])
    ax.set_ylabel(yname+yunit, fontname='Arial', fontsize=18, fontweight='bold')
    ax.set_yscale('log')
    # ax.set_title(name)

    for tick in ax.get_xticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)

    cbar = fig.colorbar(csa)
    cbar.locator = ticker.LogLocator(10)
    cbar.set_ticks(cbar.locator.tick_values(z.min(), z.max()))
    cbar.minorticks_off()
    cbar.set_label('Breakthrough (Days)', fontname='Arial', fontsize=18, fontweight='bold')
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)

    xx,yy,zz = limitcontour(x-x_corr,y/y_corr,z,xlim=templim)

    visual_levels = [1, 4, 7, 30, 365, 365*10]
    lv_lbls = ['1 d', '4 d', '1 w', '1 mo', '1 yr', '10 yr']
    ax = plt.gca()
    csb = ax.contour(xx,yy,zz,visual_levels, colors='w',locator=ticker.LogLocator(),linestyles='--',norm=LogNorm(),linewidths=1.25)
    csb.levels = lv_lbls

    ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=False)
    # ax.clabeltext(fontname='Arial')

    b1 = rho1*v1/y_corr
    b2 = rho1*v2/y_corr
    b3 = rho1*v3/y_corr

    points = [(35,b1[gf.find_nearest(x-x_corr, 35)]),(50,b2[gf.find_nearest(x-x_corr, 50)]),(65,b3[gf.find_nearest(x-x_corr, 65)])]
    v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # ax = plt.gca()
    ax.plot(x-x_corr,b1,x-x_corr,b2,x-x_corr,b3, color='k',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],textcoords='offset points',xytext=(0,40),ha='center',arrowprops=dict(facecolor='black', shrink=0.001),
    #         horizontalalignment='right', verticalalignment='bottom')
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="gray"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"), fontname='Arial', fontsize=12, fontweight='bold')
    # c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr

    # points = [(35,c1[gf.find_nearest(x-x_corr, 35)]),(50,c2[gf.find_nearest(x-x_corr, 50)]),(65,c3[gf.find_nearest(x-x_corr, 65)])]
    # v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # # ax = plt.gca()
    # ax.plot(x-x_corr,c1,x-x_corr,c2,x-x_corr,c3, color='b',linestyle='--',linewidth=1.25)
    # for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
    #         bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
    #         xytext=(0,-25), textcoords='offset points', ha='center',
    #         arrowprops=dict(arrowstyle="->"))
    c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr

    points = [(35,c1[gf.find_nearest(x-x_corr, 35)])]
    v_lbls = [str(v1)+' V (Boro)']
    # ax = plt.gca()
    ax.plot(x-x_corr,c1, color='b',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
    return


@dataclass
class Layer(object):
    """Return sum of squared errors (pred vs actual)."""

    # Invisible attribute (init-only)
    material: str = 'undefined'
    thick: float = gf.tocm(500, 'um')
    diff_type: InitVar[str] = 'undefined'

    # Initialized attribute
    temp: float = 25
    efield: float = 0
    resistivity: float = field(init=False)

    # Generated attribute
    diff: float = field(init=False)
    mob: float = field(init=False)

    area: float = 1
    res: float = field(init=False)
    volt: float = field(init=False)
    cap: float = field(init=False)
    charge: float = field(init=False)


    @property
    def resistivity(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self._res_pre*np.exp(self._res_ea/(gf.KB_EV*gf.CtoK(self.temp)))

    @resistivity.setter
    def resistivity(self, _): pass

    @property
    def diff(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self._d_0*np.exp(-self._e_a/(gf.KB_EV*gf.CtoK(self.temp)))

    @diff.setter
    def diff(self, _): pass

    @property
    def mob(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self.diff/(gf.KB_EV*gf.CtoK(self.temp))

    @mob.setter
    def mob(self, _): pass

    @property
    def volt(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self.efield * self.thick

    @volt.setter
    def volt(self, _): pass

    @property
    def res(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self.resistivity*self.thick/self.area

    @res.setter
    def res(self, _): pass

    @property
    def cap(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self._er*self.area/self.thick

    @cap.setter
    def cap(self, _): pass

    @property
    def charge(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self.cap*self.volt

    @charge.setter
    def charge(self, _): pass

    def __post_init__(self, diff_type):
        """Return layer object."""

        self._diff_type = diff_type.lower()
        try:
            self._res_pre = gf.mat_database.loc[self.material.lower(), 'pre']
        except KeyError:
            self._res_pre = gf.mat_database.loc['eva', 'pre']

        try:
            self._res_ea = gf.mat_database.loc[self.material.lower(), 'ea']
        except KeyError:
            self._res_ea = gf.mat_database.loc['eva', 'ea']

        try:
            self._er = gf.mat_database.loc[self.material.lower(), 'perm']
        except KeyError:
            self._er = gf.mat_database.loc['eva', 'perm']

        try:
            self._d_0 = gf.diff_arrh.loc[diff_type.lower(), 'pre']
        except KeyError:
            self._d_0 = gf.diff_arrh.loc['ave', 'pre']

        try:
            self._e_a = gf.diff_arrh.loc[diff_type.lower(),'ea']
        except KeyError:
            self._e_a = gf.diff_arrh.loc['ave', 'ea']

class Module:
    def __init__(self, layers=['boro','eva','sinx'], temp=25, array=100):

        self.layers=layers
        self.layer_list = [Layer(material=layer, temp=temp) for layer in layers]
        self.sys_temp=temp

        self.focus='eva'

        self.array_size = array
        self.T_1D = np.linspace(20,100,self.array_size)
        self.E_1D = np.logspace(3,6,self.array_size)
        self.T_2D,self.E_2D = np.meshgrid(self.T_1D, self.E_1D)

        self.bbt_arr = {}
        self.bbt_df = {}


    @property
    def layer_list(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._layer_list

    @layer_list.setter
    def layer_list(self,lists):
        """Return sum of squared errors (pred vs actual)."""
        self._layer_list = lists

    @property
    def module(self):
        """Return sum of squared errors (pred vs actual)."""

        return pd.DataFrame(self.layer_list,index=self.layers)

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self,layer_focus='eva'):
        self._focus = layer_focus.lower()

    @property
    def focus_df(self):
        return self.module.iloc[self.layers.index(self.focus),:]

    @property
    def focus_list(self):
        return self.layer_list[self.layers.index(self.focus)]

    @property
    def sys_temp(self):
        return self._sys_temp

    @sys_temp.setter
    def sys_temp(self,temp):
        self._sys_temp = temp
        [setattr(self.layer_list[x],'temp',temp) for x in range(len(self.layer_list))]

    @property
    def sys_volt(self):
        return self._sys_volt

    @sys_volt.setter
    def sys_volt(self,volt):
        self._sys_volt = volt
        [setattr(self.layer_list[x],'efield', self.resistivity(self.module.iloc[x,0])*volt) for x in range(len(self.layer_list))]

    @property
    def ext_volt(self):
        return self.focus_df['efield']/self.resistivity()

    @property
    def sys_res(self):
        return self.module['res'].sum()

    def thickness_adj(self,
                      layer=0,
                      thick=0,
                      by_module=False,
                      module=gf.tocm(4.55,'mm'),
                      cell=gf.tocm(200,'um'),
                      backsheet=gf.tocm(300,'um'),
                      glass='soda',
                      enc=0):
        if type(layer) is list:
            for i, lay in enumerate(layer):
                self.layer_list[lay].thick = thick[i]
        else:
            self.layer_list[layer].thick = thick

        if by_module:
            self.layer_list[enc] = (module - self.module.loc[glass, 'thick'] - cell - backsheet)/2

    def resistivity(self,focus=None,temp=None):
        if temp is not None:
            self.set_temp(temp)
        if focus is None:
            focus=self.focus
        # should be the resistivity of the layer that is to be applied against the resistance
        return self.module.resistivity[self.layers.index(focus)]/sum(self.module.thick*self.module.resistivity)

    def bbt(self,ident, layer='EVA', target=1e16, source=5e21, diff_range='Ave', diff_depth=0, thick=None):
        if thick is None:
            thick = self.module.thick[self.layers.index(layer.lower())]
        stresses_list = [Layer(material=layer,
                               diff_type=diff_range,
                               temp=temps,
                               efield=field,
                               thick=thick)
                         for temps in self.T_1D for field in self.E_1D]
        info_df = pd.DataFrame(stresses_list)

        if diff_depth == 0:
            diff_depth = thick

        info_df['depth'] = diff_depth
        # info_df['time'] = 1
        # info_df['low'] = 0.0
        # info_df['high'] = 0.0
        # info_df['changed'] = True
        ratio=target/source

        test1=c_ratio_np_df(1,info_df)
        time_bound_low = np.full(len(test1),0.0)
        time_bound_high = np.full(len(test1),0.0)
        cont = 0
        changed_logic=np.full(len(test1),True)

        secs = 0
        while cont == 0:
            if secs <= 60: # if less than a min
                delta=1 # inc by sec
            elif secs <= 3600*24: # if less than a hour
                delta=float(60) # inc by min
            elif secs <= 3600*24*30: # if less than a month
                delta=float(3600) # inc by hour
            # elif secs <= 3600*24*7: # if less than a week
            #     delta=float(3600*24) # inc by day
            elif secs <= 3600*24*365*.5: # if less than half a year
                delta=float(3600*24) # inc by day
            elif secs <= 3600*24*365: # if less than a year
                delta=float(3600*24*7) # inc by week
            elif secs <= 3600*24*365*100: # if less than 100 year
                delta=float(3600*24*14) # inc by 2 week
            elif secs > 3600*24*365*250: # if less than 250 year
                delta=float(3600*24*30) # inc by month
            elif secs > 3600*24*365*500: # if less than 500 year
                delta=float(3600*24*365) # inc by year
            elif secs > 3600*24*365*1000:
                delta=float(secs*2) # double each round

            secs+=delta
            test2=c_ratio_np_df(secs,info_df)

            time_bound_low[test1==0]=float(secs)

            comb_logic = (test2-test1)<=0.1
            test_logic = test2>=0.5
            changed_logic[comb_logic*test_logic]=False

            time_bound_high[((test2-test1)>0)*changed_logic] = float(secs)
            time_bound_low[time_bound_low>=time_bound_high] = secs-delta

            test1=test2

            if np.max(time_bound_high) < secs and np.min(time_bound_high) != 0:
                cont=1

        self.info=info_df
        # self.info['time'] = 1
        self.info['low'] = time_bound_low
        self.info['high'] = time_bound_high

        self.info['time'] = np.array([optimize.minimize_scalar(find_tauc_np_cratio_3d_df,
                                                                  bounds=(self.info['low'][y],self.info['high'][y]),
                                                                  args=(self.info.loc[y,:], ratio),
                                                                  method='bounded').x
                                                                        for y in range(len(self.info))])

        self.bbt_arr[ident] = self.info.pivot_table(values='time',columns='temp',index='efield')
        self.bbt_df[ident] = self.info

    def find_time(self, ident, temp=None, field=None, volt=None):
        if ident not in self.bbt_arr:
            print('Run Simulation first')
            return

        df = self.bbt_arr[ident]
        if temp is not None:
            self.sys_temp = df['temp'].sub(temp).abs().min()
        if volt is not None:
            self.sys_volt = df['volt'].sub(volt).abs().min()

        if field is not None:
            field = df['efield'].sub(field).abs().min()

        return df['time'][(df['temp'] == self.sys_temp) & (df['efield'] == field)]

    # def find_depth(self, time, layer='eva', diff_range='ave', target=1e16, source=5e21, temp=None, volt=None, field=None, thick=None, diff=None):

    #     if thick is None:
    #         thick = self.module['thick'][self.module['material']==layer.lower()].to_numpy()[0]
    #     stresses_list = [Layer(material=layer,
    #                            diff_type=diff_range,
    #                            temp=temps,
    #                            efield=field,
    #                            thick=thick)
    #                      for temps in self.T_1D for field in self.E_1D]

    #     df = self.module[self.module['material']==layer.lower()].to_numpy()[0]
    #     if temp is not None:
    #         self.sys_temp = df['temp'].sub(temp).abs().min()
    #     if volt is not None:
    #         self.sys_volt = df['volt'].sub(volt).abs().min()

    #     if field is not None:
    #         field = df['efield'].sub(field).abs().min()

    #     self.na_profile = c_ratio_np(depth_range,self.layer_t,(temp+273.15),self.E_layer,time,diff,mob)*source

    #     depth_fit = self.layer_t
    #     while np.count_nonzero(self.na_profile) < 0.90*self.array_size:
    #         depth_fit = 0.95*depth_fit
    #         depth_range = np.linspace(0,depth_fit,self.array_size)
    #         self.na_profile = c_ratio_np(depth_range,self.layer_t,(temp+273.15),self.E_layer,time,diff,mob)*source

    #     conc_ind = gf.find_nearest(self.na_profile,target)

    #     self.depth_diffused = depth_range[conc_ind]
    #     return self.depth_diffused

    # def find_depth_range(self,target,source,time,temp,volt,field=0):


    #     d_0_temp = self.d_0
    #     e_a_temp = self.e_a
    #     # create array from a range of diffusivities
    #     gf.diff_arrh.loc['min', 'pre']
    #     diff_min = np.log10(self.d_0*np.exp(-self.e_a/(gf.KB_EV*(temp+273.15))))
    #     self.arrhenius_data('max')
    #     diff_max = np.log10(self.d_0*np.exp(-self.e_a/(gf.KB_EV*(temp+273.15))))

    #     self.diff_range = np.logspace(diff_min,diff_max,self.array_size)

    #     self.depth_diffused_all=np.ones(self.array_size)
    #     for i in range(self.array_size):
    #         self.depth_diffused_all[i]=self.find_depth(target,source,time,temp,volt,field,self.diff_range[i])

    #     self.d_0 = d_0_temp
    #     self.e_a = e_a_temp

    #     return self.depth_diffused_all

    def leakage(self, area=None, temp=None, volt=None):

        if area is not None:
            [setattr(self.layer_list[x],'area', area) for x in range(len(self.layer_list))]

        if temp is None:
            temp = self.sys_temp
        else:
            self.sys_temp = temp
        if volt is None:
            volt = self.sys_volt
        else:
            self.sys_volt = volt

        self.I_leakage = volt/self.sys_res

        return self.I_leakage

    def find_diff(self, area=None, temp=None, volt=None):

        if area is not None:
            [setattr(self.layer_list[x],'area', area) for x in range(len(self.layer_list))]

        if temp is None:
            temp = self.sys_temp
        else:
            self.sys_temp = temp
        if volt is None:
            volt = self.sys_volt
        else:
            self.sys_volt = volt

        self.I_leakage = volt/self.sys_res

        return self.I_leakage

#%%  BTT general sim
# mod_soda=Module()

# mod_boro=Module()
# mod_boro.glass = 'Boro'
# mod_boro.resistivity()

# mod_soda.bbt('run1',layer='EVA', target=1e10, source=5e21, diff_range='Ave')


# plotcont6(mod_soda.T_1D,mod_soda.E_1D,mod_soda.bbt_arr['run1'],mod_soda.rho_layer_Tdep,mod_boro.rho_layer_Tdep,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
# plotcont8(mod_soda.T_1D,mod_soda.E_1D,mod_soda.bbt_arr['run1'],mod_soda.rho_layer_Tdep,mod_boro.rho_layer_Tdep,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])

# #%% simulation of stress testing
# # start regular module
mod_soda=Module(layers=['soda','eva','sinx'], temp=85)
mod_soda.thickness_adj(layer=[0,1,2],thick=[gf.tocm(3.2,'mm'), gf.tocm(450,'um'),gf.tocm(80,'nm')])
mod_soda.bbt('run1', layer='EVA', target=1e10, source=5e21, diff_range='max')
# # mod_soda.thickness_adj()
# # E in EVA for 1500 V and 80 C
mod_soda.sys_volt = 1000
info_soda = mod_soda.module
field_E = mod_soda.focus_df['efield']

# # generate test layout
# eset_mod = Module(layers=['soda','eva','soda'], temp=60)
# # eset_mod.rho_sinx = 3.3e16
# # eset_mod.sinx_t = gf.tocm(50,'um')
# eset_mod.thickness_adj(layer=[0,1,2],thick=[gf.tocm(2,'mm'), gf.tocm(450,'um'),gf.tocm(2,'mm')])
# eset_mod.focus_list.efield = field_E
# best_v = eset_mod.ext_volt
# eset_mod.sys_volt = 1500

# eset_curr = eset_mod.leakage(4)
# info_eset = eset_mod.module
# find V input


# find V input
# test_V = eset_mod.find_ext_vdm(temp=80,field=field_E)

# #%% simulation of stress testing
# # start regular module
# mod_soda=Module()
# # mod_soda.thickness_adj()
# # E in EVA for 1500 V and 80 C
# field_E = mod_soda.find_layer_vdm(temp=60,volt=1500,full_calcs=False)

# # generate test layout
# mod_soda_mims2 = Module()
# mod_soda_mims2.rho_sinx = 3.3e16
# # eset_mod.sinx_t = gf.tocm(50,'um')
# mod_soda_mims2.thickness_adj(module=0,glass=gf.tocm(4,'mm'),sinx=gf.tocm(100,'um'))
# # find V input


# #%% 3
# eset_mod.bbt(ident='run1',target=1e17,source=5e19,diff_range='max',diff_depth=gf.tocm(1,'um'))

# # #%% 4
# # test_time = eset_mod.find_time('run1',80,test_V)
# # test_depths = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# #%% 5
# eset_mod.bbt(ident='run2',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(2,'um'))

# # test_time = eset_mod.find_time('run2',80,test_V)
# # test_depths = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# #%% 6
# eset_mod.bbt(ident='run3',target=1e17,source=5e19,diff_range='ave',diff_depth=gf.tocm(2,'um'))

# # test_time = eset_mod.find_time('run3',80,test_V)
# # test_depths = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)


# #%% 7
# # test_depths_alt = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# #%% 8 set simulation for current diffuision
# eset_mod.bbt(ident='run1',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(2,'um'))
# eset_mod.bbt(ident='run2',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(1,'um'))
# eset_mod.bbt(ident='run3',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(1,'um'))

# mod_soda_mims2.bbt(ident='run1',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(2,'um'))
# mod_soda_mims2.bbt(ident='run2',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(1,'um'))
# mod_soda_mims2.bbt(ident='run3',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(1,'um'))
# #%% 9
# test_time_60 = eset_mod.find_time('run3',60,1500)
# test_depths_60 = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time_60,temp=80,volt=1500)
# test_diffs_60 = eset_mod.diff_range

# test_time_80 = mod_soda_mims2.find_time('run3',80,1500)
# test_depths_80_air = mod_soda_mims2.find_depth_range(target=1e17,source=5e19,time=test_time_80,temp=80,volt=1500)
# test_diffs_80_air = mod_soda_mims2.diff_range


# #%% 10
# test_time_60 = eset_mod.find_time('run1',80,test_V)
# test_depths_60 = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time_60,temp=60,volt=test_V)


# #%% 10

# eset_mod.leakage(2**2,80,1500)
# I_80=eset_mod.I_leakage_Tdep[gf.find_nearest(eset_mod.T_1D,80+273.15)]
# V_meas_80 = I_80*1.5e6
# V_meas_80_alt = I_80*680e3



# #%% 11
# mod_boro=Module()
# mod_boro.enc_pre=mod_boro.glass_pre
# mod_boro.enc_ea=mod_boro.glass_ea
# mod_boro.arrhenius_data('boro')
# mod_boro.rho_sinx=4e12
# all_glass=gf.tocm(.125,'in')
# mod_boro.thickness_adj(module=0,glass=gf.tocm(.125,'in'),enc=gf.tocm(2,'mm'),sinx=gf.tocm(100,'um'))

# mod_boro.leakage(2**2,80,test_V)
# I_80_rev=mod_boro.I_leakage_Tdep[gf.find_nearest(mod_boro.T_1D,80+273.15)]
# V_meas_80_rev = I_80*1.5e6
# V_meas_80_alt_rev = I_80*680e3


# #%% 12
# test_time_75 = eset_mod.find_time('run4',75,test_V)
# test_time_70 = eset_mod.find_time('run4',70,test_V)
# test_time_65 = eset_mod.find_time('run4',65,test_V)
# test_time_60 = eset_mod.find_time('run4',60,test_V)

# #%% 13


# mod_low_sinx=Module()
# mod_low_sinx.rho_sinx = 1e10
# mod_low_sinx.resistivity(layer_focus='sinx')
# mod_low_sinx.find_layer_vdm(80,1500)

# print(mod_low_sinx.E_layer)

# mod_high_sinx=Module()
# mod_high_sinx.rho_sinx = 1e15
# mod_high_sinx.resistivity(layer_focus='sinx')
# mod_high_sinx.find_layer_vdm(80,1500)

# print(mod_high_sinx.E_layer)
