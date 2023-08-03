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

class Module:
    def __init__(self):
        
        self.glass = 'Soda'
        self.glass_t = gf.tocm(0.125,'in')
        self.database(self.glass)
        
        self.enc = 'EVA'
        self.enc_t = gf.tocm(300,'um')
        self.database(self.enc)
        
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
        
        self.bbt_sim_arr = {}
        self.bbt_sim_df = {}
        
    def load_data(self,params):
        self.glass = params['Glass']
        self.glass_t = params['Glass Thickness']
        self.database(self.glass)

        self.enc = params['Encap']
        self.enc_t = params['Encap Thickness']
        self.database(self.enc)
        
        self.cell_t=params['Cell Thickness']
        
        self.sinx_t=params['SiNx Thickness']
        self.rho_sinx=params['SiNx Resistivity']
        
        self.diff_range = params['Diffusivity']
        self.database(self.diff_range)
        
        self.resistivity()
        
    def database(self,material):
        if material.lower() == 'boro':
            self.glass_pre = 0.04495210144020945 # std data
            self.glass_ea = 0.9702835437735396 # std data
        if material.lower() == 'soda':
            self.glass_pre = 5644.5501772317775 # std data
            self.glass_ea = 0.3590103059601377 # std data
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
            self.d_0 = 5.75e-7 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.578 # eV # mims=0.809, TOF=0.28
        if material.lower() == 'max':
            self.d_0 = 2.902e-5 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.623 # eV # mims=0.809, TOF=0.28
        if material.lower() == 'min':
            self.d_0 = 1.140e-8 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.533 # eV # mims=0.809, TOF=0.2800
    
    def resistivity(self):
        self.rho_glass = arrh(self.T_1D,self.glass_pre,self.glass_ea)
        self.rho_enc = arrh(self.T_1D,self.enc_pre,self.enc_ea)
        
        self.rho_system = self.rho_enc/(self.rho_glass*self.glass_t+self.rho_enc*self.enc_t+self.rho_sinx*self.sinx_t)
        
    def bbt(self,ident,target):
        target = 1e16/5e21 # 1e16/5e21
        # target = 1e0/5e21 # 1e16/5e21
        # target = 5e20/5e21 # 1e16/5e21
        # target = 1e15/1e16 # 1e16/5e21

        time_bound_low = np.full((self.array_size,self.array_size),0.0)
        time_bound_high = np.full((self.array_size,self.array_size),0.0)
        cont = 0

        test1 = find_tauc_np_cratio_3d_alt(np.full((self.array_size,self.array_size),1),self.enc_t,self.T_2D,self.E_2D,self.d_0,self.e_a,target,self.enc_t)
        hours = 0
        while cont == 0:
            if hours <= 3600*24:
                delta=float(60)
            elif hours <= 3600*24*30:
                delta=float(3600)
            elif hours <= 3600*24*30*365*.5:
                delta=float(3600*24)
            elif hours <= 3600*24*365*2*100:
                delta=float(3600*24*30)
            elif hours > 3600*24*365*2*200:
                delta=float(3600*24*100)
            elif hours > 3600*24*365*2*500:
                delta=float(hours*2)
            
            hours+=delta 
            
            # print(hours)
            test2 = find_tauc_np_cratio_3d_alt(np.full((self.array_size,self.array_size),hours),self.enc_t,self.T_2D,self.E_2D,self.d_0,self.e_a,target,self.enc_t)
            time_bound_low[test1==0]=float(hours)
            time_bound_high[(test2-test1)>0]=float(hours)
            time_bound_low[time_bound_low>=time_bound_high] = hours-delta
            
            test1=test2

            if np.max(time_bound_high) < hours and np.min(time_bound_high) != 0:
                cont=1

        self.bbt_sim_arr[ident] = np.array([[optimize.minimize_scalar(find_tauc_np_cratio_3d,bounds=(time_bound_low[l_E][l_T],time_bound_high[l_E][l_T]),args=(self.enc_t,self.T_1D[l_T],self.E_1D[l_E],self.d_0,self.e_a,target,self.enc_t),method='bounded').x 
                                                                        for l_T in range(self.array_size)] for l_E in range(self.array_size)])

        # arr_bt_e_vs_t = np.array(list_bt_e_vs_t)

        self.bbt_sim_df[ident] = pd.DataFrame(self.bbt_sim_arr[ident],index=self.E_1D,columns=self.T_1D)
        
        
        


#%%
Test=Module()
        
#%%
c_array_size=100

# c_d_0 = 5.75e-7 # cm/s # mims=0.021, TOF=3.41e-11
# c_e_a = 0.578 # eV # mims=0.809, TOF=0.28

c_d_0 = 2.902e-5 # cm/s # mims=0.021, TOF=3.41e-11
c_e_a = 0.623 # eV # mims=0.809, TOF=0.28

# c_d_0 = 1.140e-8 # cm/s # mims=0.021, TOF=3.41e-11
# c_e_a = 0.533 # eV # mims=0.809, TOF=0.2800

### create slices of c_ratios


arr_V_app_1D = np.linspace(2000/c_array_size,2000,c_array_size) #V 
arr_T_1D = np.linspace(20+273.15,100+273.15,c_array_size)

arr_E_1D = np.logspace(3,6,c_array_size)
arr_T_2D,arr_E_2D = np.meshgrid(arr_T_1D, arr_E_1D)

# arr_E_1D = np.logspace(3,4.247973266361806,c_array_size)

# thicnkess
c_thick_glass = gf.tocm(0.125,'in')
# c_thick_glass = gf.tocm(0.125/2,'in')
# c_thick_glass = gf.tocm(0.125*2,'in')
c_thick_cell = 180e-4
# c_thick_EVA = gf.tocm(450,'um')
c_thick_EVA = gf.tocm(300,'um')
# c_thick_EVA = gf.tocm(450*2,'um')


c_thick_SiNx = gf.tocm(80,'nm')

# resistances
boro_prefac = 0.04495210144020945 # std data
boro_ae = 0.9702835437735396 # std data
soda_prefac = 5644.5501772317775 # howard at 50 hz
soda_ae = 0.3590103059601377 # howard at 50 hz

# boro_prefac = 0.6580761914650776 # my data
# boro_ae = 0.8507365332956724 # my data
# soda_prefac = 0.023546001890093236 # my data
# soda_ae = 0.9471251012618027 # my data

EVA_prefac = 19863.639619529386
EVA_ae = 0.6319537614631568

#%%
arr_rho_boro = arrh(arr_T_1D,boro_prefac,boro_ae)
arr_rho_soda = arrh(arr_T_1D,soda_prefac,soda_ae)
arr_rho_EVA = arrh(arr_T_1D,EVA_prefac,EVA_ae)
arr_rho_SiNx = 1e13 #3e13
# arr_rho_SiNx = 1e10 #3e13
# arr_rho_SiNx = 1e15 #3e13

arr_rho_system_1D_boro = arr_rho_EVA/(arr_rho_boro*c_thick_glass+arr_rho_EVA*c_thick_EVA+arr_rho_SiNx*c_thick_SiNx)
arr_rho_system_1D_soda = arr_rho_EVA/(arr_rho_soda*c_thick_glass+arr_rho_EVA*c_thick_EVA+arr_rho_SiNx*c_thick_SiNx)
# arr_rho_system_1D_soda = arr_rho_EVA/(5e12*c_thick_glass+arr_rho_EVA*c_thick_EVA+arr_rho_SiNx*c_thick_SiNx)



arr_rho_system_2D,arr_V_app_2D = np.meshgrid(arr_rho_system_1D_soda, arr_V_app_1D)

arr_E_EVA_2D = arr_rho_system_2D*arr_V_app_2D

# plotcont1(arr_T_1D,arr_V_app_1D,arr_E_EVA_2D*1e-6,'Variance in E for V & T',273.15,yname='V app',levels=100)


# %%
# df_E_EVA=pd.DataFrame(arr_E_EVA_2D,index=arr_V_app_1D,columns=arr_T_1D)

c_target = 1e16/5e21 # 1e16/5e21
# c_target = 1e0/5e21 # 1e16/5e21
# c_target = 5e20/5e21 # 1e16/5e21
# c_target = 1e15/1e16 # 1e16/5e21

time_bound_low = np.full((c_array_size,c_array_size),0.0)
time_bound_high = np.full((c_array_size,c_array_size),0.0)
cont = 0

test1 = find_tauc_np_cratio_3d_alt(np.full((c_array_size,c_array_size),1),c_thick_EVA,arr_T_2D,arr_E_2D,c_d_0,c_e_a,c_target,c_thick_EVA)
hours = 0
while cont == 0:
    if hours <= 3600*24:
        delta=float(60)
    elif hours <= 3600*24*30:
        delta=float(3600)
    elif hours <= 3600*24*30*365*.5:
        delta=float(3600*24)
    elif hours <= 3600*24*365*2*100:
        delta=float(3600*24*30)
    elif hours > 3600*24*365*2*200:
        delta=float(3600*24*100)
    elif hours > 3600*24*365*2*500:
        delta=float(hours*2)
    
    hours+=delta 
    
    # print(hours)
    test2 = find_tauc_np_cratio_3d_alt(np.full((c_array_size,c_array_size),hours),c_thick_EVA,arr_T_2D,arr_E_2D,c_d_0,c_e_a,c_target,c_thick_EVA)
    time_bound_low[test1==0]=float(hours)
    time_bound_high[(test2-test1)>0]=float(hours)
    time_bound_low[time_bound_low>=time_bound_high] = hours-delta
    
    test1=test2

    if np.max(time_bound_high) < hours and np.min(time_bound_high) != 0:
        cont=1

list_bt_e_vs_t = [[optimize.minimize_scalar(find_tauc_np_cratio_3d,bounds=(time_bound_low[l_E][l_T],time_bound_high[l_E][l_T]),args=(c_thick_EVA,arr_T_1D[l_T],arr_E_1D[l_E],c_d_0,c_e_a,c_target,c_thick_EVA),method='bounded').x 
                                                                for l_T in range(c_array_size)] for l_E in range(c_array_size)]

arr_bt_e_vs_t = np.array(list_bt_e_vs_t)

df_bt_e_vs_t=pd.DataFrame(arr_bt_e_vs_t,index=arr_E_1D,columns=arr_T_1D)

#%%
plotcont6(arr_T_1D,arr_E_1D,arr_bt_e_vs_t,arr_rho_system_1D_soda,arr_rho_system_1D_boro,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
plotcont8(arr_T_1D,arr_E_1D,arr_bt_e_vs_t,arr_rho_system_1D_soda,arr_rho_system_1D_boro,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])


# %%

var_temp_ind = gf.find_nearest(arr_T_1D,85+273.15)
var_v = 1500
var_rho = arr_rho_system_1D_soda[var_temp_ind]
var_E_ind= gf.find_nearest(arr_E_1D,var_rho*var_v) 

var_btt = arr_bt_e_vs_t[var_E_ind,var_temp_ind]

print('Time = ',var_btt/(3600*24))
#%%

var_rho_boro = arrh(25+273.15,boro_prefac,boro_ae)
var_rho_soda = arrh(25+273.15,soda_prefac,soda_ae)
var_rho_EVA = arrh(25+273.15,EVA_prefac,EVA_ae)
