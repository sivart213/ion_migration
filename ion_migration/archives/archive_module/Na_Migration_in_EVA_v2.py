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


def plotcont7(x,y,z,rho1,v1,v2,v3,name='',y_corr=1,xname='Time [D]',yname='E',ind=0,levels=50,templim=[0,7]):
    
    if y_corr == 1e6:
        yunit = ' [MV/cm]'
    else:
        yunit = ' [V/cm]'
    
    x=x/(3600*24)

    
    fig, ax = plt.subplots()
    csa = ax.contourf(x,y/y_corr,z,np.logspace(np.log10(z.min()),np.log10(z.max()), levels), locator = ticker.LogLocator(),  cmap='gist_heat')
    
    ax.set_xlabel(xname)
    ax.set_xlim(templim[0],templim[1])
    ax.set_ylabel(yname+yunit)
    ax.set_yscale('log')
    ax.set_title(name)
    
    cbar = fig.colorbar(csa)
    cbar.locator = ticker.LogLocator(10)
    cbar.set_ticks(cbar.locator.tick_values(z.min(), z.max()))
    cbar.minorticks_off()
    cbar.set_label('Diffusivity [cm$^{2}$/s]')
    
   
    xx,yy,zz = limitcontour(x,y/y_corr,z,xlim=templim)
    
    visual_levels = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10]
    lv_lbls = ['$10^{-16}$', '$10^{-15}$', '$10^{-14}$', '$10^{-13}$', '$10^{-12}$', '$10^{-11}$', '$10^{-10}$']
    ax = plt.gca()
    csb = ax.contour(xx,yy,zz,visual_levels, colors='w',locator=ticker.LogLocator(),linestyles='--',norm=LogNorm(),linewidths=1.25)
    csb.levels = lv_lbls

    ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=False)
        
    b1 = rho1*v1/y_corr*np.ones_like(x)
    b2 = rho1*v2/y_corr*np.ones_like(x)
    b3 = rho1*v3/y_corr*np.ones_like(x)
    
    points = [(1,b1[0]),(2.5,b2[0]),(4,b3[0])]
    v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # ax = plt.gca()
    ax.plot(x,b1,x,b2,x,b3, color='k',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],textcoords='offset points',xytext=(0,40),ha='center',arrowprops=dict(facecolor='black', shrink=0.001),
    #         horizontalalignment='right', verticalalignment='bottom')
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="gray"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
    
    return

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
    # c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr
    
    # points = [(35,c1[gf.find_nearest(x-x_corr, 35)])]
    # v_lbls = [str(v1)+' V (Boro)']
    # # ax = plt.gca()
    # ax.plot(x-x_corr,c1, color='b',linestyle='--',linewidth=1.25)
    # for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
    #         bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
    #         xytext=(0,25), textcoords='offset points', ha='center',
    #         arrowprops=dict(arrowstyle="->"))
    return



#%%
c_array_size=100



c_d_0 = 5.75e-7 # cm/s # mims=0.021, TOF=3.41e-11
c_e_a = 0.578 # eV # mims=0.809, TOF=0.28

# c_d_0 = 1.40e-5 # cm/s # mims=0.021, TOF=3.41e-11
# c_e_a = 0.623 # eV # mims=0.809, TOF=0.28


# updated
# c_d_0 = 0.277 # cm/s # mims=0.277, TOF=2.157e-9
# c_e_a = 0.928 # eV # mims=0.928, TOF=0.4524
# c_d_0 = 2.157e-9 # cm/s # mims=0.277, TOF=2.157e-9
# c_e_a = 0.4524 # eV # mims=0.928, TOF=0.4524
# updated2
# c_d_0 = 14.31 # cm/s # mims=0.277, TOF=2.157e-9
# c_e_a = 1.052 # eV # mims=0.928, TOF=0.4524
# c_d_0 = 1.847 # cm/s # mims=0.277, TOF=2.157e-9
# c_e_a = 0.422 # eV # mims=0.928, TOF=0.4524
# fake
# c_d_0 = 1.33e-5 # cm/s # mims=0.277, TOF=2.157e-9
# c_e_a = gf.KB_EV*(145+273.15) # eV # mims=0.928, TOF=0.4524
### create slices of c_ratios


arr_V_app_1D = np.linspace(2000/c_array_size,2000,c_array_size) #V 
arr_T_1D = np.linspace(20+273.15,100+273.15,c_array_size)
# arr_T_1D = np.linspace(100/c_array_size+273.15,100+273.15,c_array_size)
# arr_E_1D = np.logspace(3,4.247973266361806,c_array_size)
arr_E_1D = np.logspace(3,6,c_array_size)
arr_T_2D,arr_E_2D = np.meshgrid(arr_T_1D, arr_E_1D)




#Resistivities raw
Glass_250 = 1e8 #ohm-cm at 250C
Glass_350 = 1*10**(6.5) #ohm-cm at 350C

# EVA_9100 = 1.1e15 #ohm-cm at 25C (assumed)
# EVA_val = 1e13

# SiNx = 5e10 #ohm-cm


# thicnkess
c_thick_glass = gf.tocm(0.125,'in')
c_thick_cell = 180e-4
c_thick_EVA = gf.tocm(450,'um')
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
arr_rho_SiNx = 3e13 #3e13

arr_rho_system_1D_boro = arr_rho_EVA/(arr_rho_boro*c_thick_glass+arr_rho_EVA*c_thick_EVA+arr_rho_SiNx*c_thick_SiNx)
arr_rho_system_1D_soda = arr_rho_EVA/(arr_rho_soda*c_thick_glass+arr_rho_EVA*c_thick_EVA+arr_rho_SiNx*c_thick_SiNx)
# arr_rho_system_1D_soda = arr_rho_EVA/(5e12*c_thick_glass+arr_rho_EVA*c_thick_EVA+arr_rho_SiNx*c_thick_SiNx)



arr_rho_system_2D,arr_V_app_2D = np.meshgrid(arr_rho_system_1D_soda, arr_V_app_1D)

arr_E_EVA_2D = arr_rho_system_2D*arr_V_app_2D

# plotcont1(arr_T_1D,arr_V_app_1D,arr_E_EVA_2D*1e-6,'Variance in E for V & T',273.15,yname='V app',levels=100)


# %%
df_E_EVA=pd.DataFrame(arr_E_EVA_2D,index=arr_V_app_1D,columns=arr_T_1D)

c_target = 1e16/5e21

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

plotcont6(arr_T_1D,arr_E_1D,arr_bt_e_vs_t,arr_rho_system_1D_soda,arr_rho_system_1D_boro,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
plotcont8(arr_T_1D,arr_E_1D,arr_bt_e_vs_t,arr_rho_system_1D_soda,arr_rho_system_1D_boro,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
# 

# #%%
# time1=np.array(range(0,3600*24*7,3600))
# time1[0]=60

# c_rho_boro = arrh(85+273.15,boro_prefac,boro_ae)
# c_rho_soda = arrh(85+273.15,soda_prefac,soda_ae)
# c_rho_EVA = arrh(85+273.15,EVA_prefac,EVA_ae)
# c_rho_SiNx = 3e13 #3e13

# rho_tot = c_rho_EVA/(c_rho_soda*c_thick_glass+c_rho_EVA*c_thick_EVA+c_rho_SiNx*c_thick_SiNx)

# # result = np.array([10**optimize.minimize_scalar(c_np,bracket=(np.log10(1e-11),np.log10(1e-9)),args=(c_thick_EVA,5e17,1e16,c_thick_EVA,85+273.15,1.000e+03,y),method='Golden').x for y in range(60,3600*24*7,3600)])

# diff_needed = np.array([[10**optimize.minimize_scalar(c_np,bracket=(np.log10(1e-15),np.log10(1e-9)),args=(c_thick_EVA,2e12,1e15,c_thick_EVA,25+273.15,E,y),method='Golden').x for y in time1] for E in (arr_E_1D)])

# # result = fsolve(c_np,1e-13,args=(c_thick_EVA,5e17,1e12,c_thick_EVA,85+273.15,e_assumed,3600*24*4)) bounds=(-15,-12),
# plotcont7(time1,arr_E_1D,diff_needed,rho_tot,1500,1000,600,'Variance in Diffusivity for E & t',1e6,levels=25)

# #%%
# arr_depth=gf.tocm(np.linspace(1,500,c_array_size),'um')

# c_diff = c_d_0*np.exp(-c_e_a/(gf.KB_EV*(145+273.15)))
# c_mob = c_diff/(gf.KB_EV*(145+273.15))

# test=c_ratio_np(arr_depth,c_thick_EVA,(145+273.15),0,(60),c_diff,c_mob)*1e19
    
