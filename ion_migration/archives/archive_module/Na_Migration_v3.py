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

def arrh(T,pre_fac,E_A):
    return pre_fac*np.exp(E_A/(gf.KB_EV*T))

def c_ratio_np(depth,thick,temp,e_app,time,diff,mob):
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (1/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))

def c_np(diff,depth,c_f,c_0,thick,temp,e_app,time):
    # if diff <0:
    #     diff=10**diff
    diff=10**diff
    mob = diff/(gf.KB_EV*temp)
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return ((c_0/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))-c_f)**2

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

def find_diff_np(diff, time, depth, temp, e_app, target=0.08, thick = 450e-4):
    if diff < 0:
        diff = 10**diff

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

def plotcont8(x, y, z, rho1, rho2, v1, v2, v3, name='', x_corr=0, y_corr=1,
              xname='Temperature', yname='E', ind=0, levels=50, templim=[25, 85]):

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
    csa = ax.contourf(x-x_corr, y/y_corr, z,
                      np.logspace(np.log10(z.min()), np.log10(z.max()), levels),
                      locator = ticker.LogLocator(),
                      cmap='magma')



    ax.set_xlabel(xname+xunit, fontname='Arial', fontsize=18, fontweight='bold')
    ax.set_xlim(templim[0],templim[1])
    ax.set_ylabel(yname+yunit, fontname='Arial', fontsize=18, fontweight='bold')
    ax.set_yscale('log')
    # ax.set_title(name)

    for tick in ax.get_xticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(14)

    cbar = fig.colorbar(csa)
    cbar.locator = ticker.LogLocator(10)
    cbar.set_ticks(cbar.locator.tick_values(z.min(), z.max()))
    cbar.minorticks_off()
    cbar.set_label('Breakthrough (Days)', fontname='Arial', fontsize=18, fontweight='bold')
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(14)

    xx,yy,zz = limitcontour(x-x_corr,y/y_corr,z,xlim=templim)
    b_lvls = [1, 4, 7, 30, 365, 365*10]

    b_pts = [(xx[int(xx.argmax()-10)], yy[gf.find_nearest(zz[:, -10], x)])
             if gf.find_nearest(zz[:, -10], x) != 0
             else (xx[gf.find_nearest(zz[10, :], x)], yy[int(yy.argmin()+10)])
             for x in b_lvls]

# b1[gf.find_nearest(x-x_corr, 35)]
    b_lbls = ['1 d', '4 d', '1 w', '1 mo', '1 yr', '10 yr']

    while b_pts[0][1] > yy[int(yy.argmax()-5)]:
        b_pts.remove(b_pts[0])
        b_lbls.remove(b_lbls[0])
        b_lvls.remove(b_lvls[0])

    ax = plt.gca()
    csb = ax.contour(xx, yy, zz, b_lvls, colors='w', locator=ticker.LogLocator(),
                     linestyles=':', norm=LogNorm(), linewidths=1.25)
    csb.levels = b_lbls

    ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=b_pts)

    #Soda 1
    rho1_2d, y_2d = np.meshgrid(rho1,y)
    volt1 = y_2d/rho1_2d
    xx,yy,zz = limitcontour(x-x_corr,y/y_corr,volt1,xlim=templim)

    c_pts = [(55,yy[gf.find_nearest(volt1[:,gf.find_nearest(x-x_corr, 55)], v3)]),
              (45,yy[gf.find_nearest(volt1[:,gf.find_nearest(x-x_corr, 45)], v2)]),
              (35,yy[gf.find_nearest(volt1[:,gf.find_nearest(x-x_corr, 35)], v1)])]
    c_lvls = [v3,v2,v1]
    c_lbls = [str(v3*1e-3)+' kV',str(v2*1e-3)+' kV',str(v1*1e-3)+' kV']

    ax = plt.gca()
    csc = ax.contour(xx, yy, zz, c_lvls,  colors='k', locator=ticker.LogLocator(),
                     linestyles='--', norm=LogNorm(), linewidths=1.25)
    csc.levels = c_lbls
    try:
        ax.clabel(csc, csc.levels, inline=True, fontsize=14, manual=c_pts)
    except IndexError:
        pass

    #boro 1
    rho2_2d, y_2d = np.meshgrid(rho2,y)
    volt2 = y_2d/rho2_2d


    xx,yy,zz = limitcontour(x-x_corr,y/y_corr,volt2,xlim=templim)

    point = [(40,yy[gf.find_nearest(volt2[:,gf.find_nearest(x-x_corr, 40)], v1)])]
    d_lvls = [v1]
    d_lbls = [str(v1*1e-3)+' kV (boro)']
    ax = plt.gca()
    csd = ax.contour(xx,yy,zz,d_lvls, colors='k',locator=ticker.LogLocator(),linestyles='-.',norm=LogNorm(),linewidths=1.25)
    csd.levels = d_lbls

    try:
        ax.clabel(csd, csd.levels, inline=True, fontsize=14, manual=point)
    except IndexError:
        pass

    ax.tick_params(direction='in', length=6, width=2)

    [x.set_linewidth(2) for x in ax.spines.values()]
    ax.minorticks_off()
    plt.tight_layout()

    return
#%%
class Module:
    def __init__(self):
        self.glass = 'Soda'
        self.glass_t = gf.tocm(0.125,'in')


        self.enc = 'EVA'
        self.enc_t = gf.tocm(450,'um')


        self.diff_range = 'Ave'
        self.database(self.diff_range)

        self.cell_t = 180e-4
        self.sinx_t = gf.tocm(80,'nm')

        self.rho_sinx = 1e13


        self.array_size = 100
        self.T_1D = np.linspace(20+273.15,100+273.15,self.array_size)
        self.E_1D = np.logspace(3,6,self.array_size)
        self.T_2D,self.E_2D = np.meshgrid(self.T_1D, self.E_1D)

        self.resistivity()

        self.bbt_arr = {}
        self.bbt_df = {}


    def database(self,material):
        if material.lower() == 'boro':
            self.glass_pre = 0.04495210144020945 # std data
            self.glass_ea = 0.9702835437735396 # std data
        if material.lower() == 'soda':
            self.glass_pre = 0.00570168 # std data :5644.5501772317775
            self.glass_ea = 1.16717 # std data: 0.3590103059601377
        if material.lower() == 'boro alt':
            self.glass_pre = 0.6580761914650776 # my data
            self.glass_ea = 0.8507365332956724 # my data
        if material.lower() == 'soda alt':
            self.glass_pre = 0.023546001890093236 # my data
            self.glass_ea = 0.9471251012618027 # my data
        if material.lower() == 'eva':
            self.enc_pre = 19863.639619529386
            self.enc_ea = 0.6319537614631568
        if material.lower() == 'ave':
            self.d_0 = 8.038e-8 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.544 # eV # mims=0.809, TOF=0.28
        if material.lower() == 'max':
            self.d_0 = 8.52e-7 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.6 # eV # mims=0.809, TOF=0.28
        if material.lower() == 'min':
            self.d_0 = 2.03e-8 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.53 # eV # mims=0.809, TOF=0.2800
        if material.lower() == 'tof':
            self.d_0 = 6.953e-8 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.533 # eV # mims=0.809, TOF=0.2800
        # if material.lower() == 'air':
        #     d


    def thickness_adj(self, module = gf.tocm(4.55,'mm'),cell = gf.tocm(200,'um'),backsheet = gf.tocm(300,'um'),glass=0,enc=0,sinx=0):
        if glass!=0:
            self.glass_t = glass

        if enc!=0:
            self.enc_t = enc
        elif module != 0 and enc == 0:
            self.enc_t = (module - self.glass_t - cell - backsheet)/2

        if sinx!=0:
            self.sinx = sinx

        self.resistivity()


    def focus(self,layer_focus='EVA'):
        self.layer_material = layer_focus

        if layer_focus.lower() == 'eva':
            self.layer_rho = self.rho_enc
            self.layer_t = self.enc_t
        elif layer_focus.lower() == 'glass':
            self.layer_rho = self.rho_glass
            self.layer_t = self.glass_t
        elif layer_focus.lower() == 'sinx':
            self.layer_rho = self.rho_sinx
            self.layer_t = self.sinx_t
        else:
            print('No target layer for system')
            self.layer_rho = 1e10
            self.layer_t = 1e-4


    def resistivity(self,layer_focus='EVA'):
        self.database(self.glass)
        self.rho_glass = arrh(self.T_1D,self.glass_pre,self.glass_ea)

        self.database(self.enc)
        self.rho_enc = arrh(self.T_1D,self.enc_pre,self.enc_ea)



        self.focus(layer_focus)
        # should be the resistivity of the layer that is to be applied against the resistance
        self.rho_layer_Tdep = self.layer_rho/(self.rho_glass*self.glass_t+self.rho_enc*self.enc_t+self.rho_sinx*self.sinx_t)


    def bbt(self,ident,layer_focus='EVA',target=1e16,source=5e21,diff_range='Ave',diff_depth=0):

        self.focus(layer_focus)

        if diff_depth == 0:
            diff_depth = self.layer_t

        self.database(diff_range)

        ratio=target/source

        time_bound_low = np.full((self.array_size,self.array_size),0.0)
        time_bound_high = np.full((self.array_size,self.array_size),0.0)
        cont = 0
        changed_logic=np.full((self.array_size,self.array_size),True)

        test1 = find_tauc_np_cratio_3d_alt(np.full((self.array_size,self.array_size),1),diff_depth,self.T_2D,self.E_2D,self.d_0,self.e_a,ratio,self.layer_t)
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

            # print(secs)
            test2 = find_tauc_np_cratio_3d_alt(np.full((self.array_size,self.array_size),secs),diff_depth,self.T_2D,self.E_2D,self.d_0,self.e_a,ratio,self.layer_t)
            time_bound_low[test1==0]=float(secs)

            comb_logic = (test2-test1)<=0.1
            test_logic = test2>=0.5
            changed_logic[comb_logic*test_logic]=False

            time_bound_high[((test2-test1)>0)*changed_logic]=float(secs)
            time_bound_low[time_bound_low>=time_bound_high] = secs-delta

            test1=test2

            if np.max(time_bound_high) < secs and np.min(time_bound_high) != 0:
                cont=1

        self.bbt_arr[ident] = np.array([[optimize.minimize_scalar(find_tauc_np_cratio_3d,bounds=(time_bound_low[l_E][l_T],time_bound_high[l_E][l_T]),args=(diff_depth,self.T_1D[l_T],self.E_1D[l_E],self.d_0,self.e_a,ratio,self.layer_t),method='bounded').x
                                                                        for l_T in range(self.array_size)] for l_E in range(self.array_size)])

        self.bbt_df[ident] = pd.DataFrame(self.bbt_arr[ident],index=self.E_1D,columns=self.T_1D)



    def find_layer_vdm(self,temp,volt,full_calcs=True):
        self.E_layer = self.rho_layer_Tdep[gf.find_nearest(self.T_1D,temp+273.15)]*volt

        if full_calcs:
            self.rho_vdm = self.rho_layer_Tdep[gf.find_nearest(self.T_1D,temp+273.15)]
            self.V_layer = self.E_layer*self.layer_t

            self.E_layer_Tdep = self.rho_layer_Tdep*volt
            self.V_layer_Tdep = self.E_layer_Tdep*self.layer_t
        return self.E_layer

    def find_ext_vdm(self,temp,field):
        self.V_ext = field/self.rho_layer_Tdep[gf.find_nearest(self.T_1D,temp+273.15)]
        self.V_ext_Tdep = field/self.rho_layer_Tdep
        return self.V_ext

    def find_time(self,ident,temp,volt,field=0):
        if ident not in self.bbt_arr:
            print('Run Simulation first')
            return

        if field!=0:
            self.E_layer=field
        else:
            self.find_layer_vdm(temp,volt,full_calcs=False)

        self.btt_found = self.bbt_arr[ident][gf.find_nearest(self.E_1D,self.E_layer),gf.find_nearest(self.T_1D,temp+273.15)]
        return self.btt_found

    def find_depth(self,target,source,time,temp,volt,field=0,diff=0):
        if field!=0:
            self.E_layer=field
        else:
            self.find_layer_vdm(temp,volt,full_calcs=False)

        if diff==0:
            diff = self.d_0*np.exp(-self.e_a/(gf.KB_EV*(temp+273.15)))
        mob = diff/(gf.KB_EV*(temp+273.15))

        depth_range = np.linspace(0,self.layer_t,self.array_size)
        self.na_profile = c_ratio_np(depth_range,self.layer_t,(temp+273.15),self.E_layer,time,diff,mob)*source

        depth_fit = self.layer_t
        while np.count_nonzero(self.na_profile) < 0.90*self.array_size:
            depth_fit = 0.95*depth_fit
            depth_range = np.linspace(0,depth_fit,self.array_size)
            self.na_profile = c_ratio_np(depth_range,self.layer_t,(temp+273.15),self.E_layer,time,diff,mob)*source

        conc_ind = gf.find_nearest(self.na_profile,target)

        self.depth_diffused = depth_range[conc_ind]
        return self.depth_diffused

    def find_depth_range(self,target,source,time,temp,volt,field=0):
        d_0_temp = self.d_0
        e_a_temp = self.e_a

        self.database('min')
        diff_min = np.log10(self.d_0*np.exp(-self.e_a/(gf.KB_EV*(temp+273.15))))
        self.database('max')
        diff_max = np.log10(self.d_0*np.exp(-self.e_a/(gf.KB_EV*(temp+273.15))))
        self.diff_range = np.logspace(diff_min,diff_max,self.array_size)

        self.depth_diffused_all=np.ones(self.array_size)
        for i in range(self.array_size):
            self.depth_diffused_all[i]=self.find_depth(target,source,time,temp,volt,field,self.diff_range[i])

        self.d_0 = d_0_temp
        self.e_a = e_a_temp

        return self.depth_diffused_all

    def leakage(self,area,temp,volt):

        self.R_glass_Tdep = self.rho_glass*self.glass_t/area
        self.R_enc_Tdep = self.rho_enc*self.enc_t/area
        self.R_sinx_Tdep = self.rho_sinx*self.sinx_t/area

        self.R_system_Tdep = self.R_glass_Tdep + self.R_enc_Tdep + self.R_sinx_Tdep


        self.R_system = self.R_system_Tdep[gf.find_nearest(self.T_1D,temp+273.15)]

        self.I_leakage_Tdep = volt/self.R_system_Tdep

        self.I_leakage = volt/self.R_system

        return self.I_leakage

    def find_diff(self,time,temp,volt=1000,cell=1e10,source=5e21):

        e_app = self.find_layer_vdm(temp,volt,full_calcs=False)
        target = cell/source
        diff = optimize.least_squares(find_diff_np,-15,jac='3-point',bounds=(-18, -10),args=(time, self.layer_t, temp+273.15, e_app, target, self.layer_t))['x']


        return diff
#%%  BTT general sim
mod_soda=Module()
mod_soda.thickness_adj(glass=gf.tocm(3.2,'mm'),enc=gf.tocm(300,'um'))

mod_boro=Module()
mod_boro.glass = 'Boro'
mod_soda.thickness_adj(glass=gf.tocm(3.2,'mm'),enc=gf.tocm(300,'um'))
mod_boro.resistivity()

mod_soda.bbt('run1',layer_focus='EVA',target=1e10,source=5e21,diff_range='Max')

# diff = mod_soda.find_diff(96*3600, 80, 1000,1e10,5e21)
#%%
# temp = 85
# volt = 1500
# hours = 2.5
# thick = 450

# mod_soda=Module()
# mod_soda.thickness_adj(glass=gf.tocm(3.2,'mm'),enc=gf.tocm(thick,'um'))
# mod_soda.resistivity()


# e_app =  mod_soda.find_layer_vdm(temp,volt,full_calcs=False)

# diff_needed = optimize.minimize_scalar(c_np,bracket=(np.log10(1e-15),np.log10(1e-9)),args=(gf.tocm(thick,'um'),1e10,5e21,gf.tocm(thick,'um'),temp+273.15,e_app,hours*3600),method='Golden').x

# diff_needed = 10**diff_needed
#%%
# plotcont6(mod_soda.T_1D,mod_soda.E_1D,mod_soda.bbt_arr['run1'],mod_soda.rho_layer_Tdep,mod_boro.rho_layer_Tdep,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
plotcont8(mod_soda.T_1D, mod_soda.E_1D, mod_soda.bbt_arr['run1'],
          mod_soda.rho_layer_Tdep, mod_boro.rho_layer_Tdep,
          1500, 1000, 600, 'Variance in Breakthrough Time for E & T',
          273.15, 1e6, levels=25, templim=[25, 100])



# #%% simulation of stress testing
# # start regular module
# mod_soda=Module()
# # mod_soda.thickness_adj()
# # E in EVA for 1500 V and 80 C
# field_E = mod_soda.find_layer_vdm(temp=80,volt=1500,full_calcs=False)

# # generate test layout
# mod_soda_mims = Module()
# # mod_soda_mims.rho_sinx = 3.3e16
# # mod_soda_mims.sinx_t = gf.tocm(50,'um')
# mod_soda_mims.thickness_adj(module=0,glass=gf.tocm(4,'mm'),sinx=1e-30)
# # find V input
# test_V = mod_soda_mims.find_ext_vdm(temp=80,field=field_E)

# #%% simulation of stress testing
# # start regular module
# mod_soda=Module()
# # mod_soda.thickness_adj()
# # E in EVA for 1500 V and 80 C
# field_E = mod_soda.find_layer_vdm(temp=80,volt=1500,full_calcs=False)

# # generate test layout
# mod_soda_mims2 = Module()
# mod_soda_mims2.rho_sinx = 3.3e16
# # mod_soda_mims.sinx_t = gf.tocm(50,'um')
# mod_soda_mims2.thickness_adj(module=0,glass=gf.tocm(4,'mm'),sinx=gf.tocm(100,'um'))
# # find V input


# #%% 3
# mod_soda_mims.bbt(ident='run1',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(1,'um'))

# #%% 4
# test_time = mod_soda_mims.find_time('run1',80,test_V)
# test_depths = mod_soda_mims.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# #%% 5
# mod_soda_mims.bbt(ident='run2',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(2,'um'))

# test_time = mod_soda_mims.find_time('run2',80,test_V)
# test_depths = mod_soda_mims.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# #%% 6
# mod_soda_mims.bbt(ident='run3',target=1e17,source=5e19,diff_range='ave',diff_depth=gf.tocm(2,'um'))

# test_time = mod_soda_mims.find_time('run3',80,test_V)
# test_depths = mod_soda_mims.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)


# #%% 7
# test_depths_alt = mod_soda_mims.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# #%% 8 set simulation for current diffuision
# mod_soda_mims.bbt(ident='run1',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(2,'um'))
# mod_soda_mims.bbt(ident='run2',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(1,'um'))
# mod_soda_mims.bbt(ident='run3',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(1,'um'))

# mod_soda_mims2.bbt(ident='run1',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(2,'um'))
# mod_soda_mims2.bbt(ident='run2',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(1,'um'))
# mod_soda_mims2.bbt(ident='run3',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(1,'um'))
# #%% 9
# test_time_80 = mod_soda_mims.find_time('run1',60,1500)
# test_depths_80 = mod_soda_mims.find_depth_range(target=1e17,source=5e19,time=test_time_80,temp=80,volt=1500)
# test_diffs_80 = mod_soda_mims.diff_range

# test_time_80 = mod_soda_mims2.find_time('run1',80,1500)
# test_depths_80_air = mod_soda_mims2.find_depth_range(target=1e17,source=5e19,time=test_time_80,temp=80,volt=1500)
# test_diffs_80_air = mod_soda_mims2.diff_range


# #%% 10
# test_time_60 = mod_soda_mims.find_time('run1',80,test_V)
# test_depths_60 = mod_soda_mims.find_depth_range(target=1e17,source=5e19,time=test_time_60,temp=60,volt=test_V)


# #%% 10

# mod_soda_mims.leakage(2**2,80,1500)
# I_80=mod_soda_mims.I_leakage_Tdep[gf.find_nearest(mod_soda_mims.T_1D,80+273.15)]
# V_meas_80 = I_80*1.5e6
# V_meas_80_alt = I_80*680e3



# #%% 11
# mod_boro=Module()
# mod_boro.enc_pre=mod_boro.glass_pre
# mod_boro.enc_ea=mod_boro.glass_ea
# mod_boro.database('boro')
# mod_boro.rho_sinx=4e12
# all_glass=gf.tocm(.125,'in')
# mod_boro.thickness_adj(module=0,glass=gf.tocm(.125,'in'),enc=gf.tocm(2,'mm'),sinx=gf.tocm(100,'um'))

# mod_boro.leakage(2**2,80,test_V)
# I_80_rev=mod_boro.I_leakage_Tdep[gf.find_nearest(mod_boro.T_1D,80+273.15)]
# V_meas_80_rev = I_80*1.5e6
# V_meas_80_alt_rev = I_80*680e3


# #%% 12
# test_time_75 = mod_soda_mims.find_time('run4',75,test_V)
# test_time_70 = mod_soda_mims.find_time('run4',70,test_V)
# test_time_65 = mod_soda_mims.find_time('run4',65,test_V)
# test_time_60 = mod_soda_mims.find_time('run4',60,test_V)

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
