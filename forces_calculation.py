import numpy as np
from numpy import cos, sin
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.default'] = 'regular'
linewidth_major=2
linewidth_minor=1
def scatter_point(x, y, **kwargs):
    default_kwargs = dict(s=30, color='black', zorder=5)
    default_kwargs.update(kwargs)
    return plt.scatter(x, y, **default_kwargs)

def normal_force(location_x_top, location_x_bottom, pressure_top, pessure_bottom):
    diff_top = np.array([location_x_top[i] - location_x_top[i-1] for i in range(1, len(location_x_top))])
    diff_bottom = np.array([location_x_bottom[i] - location_x_bottom[i-1] for i in range(1, len(location_x_bottom))])
    avg_pressure_top=np.array([(pressure_top[i] + pressure_top[i-1])/2 for i in range(1, len(pressure_top))])
    avg_pressure_bottom=np.array([(pessure_bottom[i] + pessure_bottom[i-1])/2 for i in range(1, len(pessure_bottom))])
    force=0
    for length, pressure in zip(diff_top,avg_pressure_top):
        force-=length*pressure  
    for length, pressure in zip(diff_bottom,avg_pressure_bottom):
        force+=length*pressure  
    return force

def tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom):
    diff_y_top = np.array([location_y_top[i] - location_y_top[i-1] for i in range(1, len(location_y_top))])
    diff_y_bottom= np.array([-location_y_bottom[i] + location_y_bottom[i-1] for i in range(1, len(location_y_bottom))])
    avg_pressure_top=np.array([(pressure_top[i] + pressure_top[i-1])/2 for i in range(1, len(pressure_top))])
    avg_pressure_bottom=np.array([(pessure_bottom[i] + pessure_bottom[i-1])/2 for i in range(1, len(pessure_bottom))])
    force=0
    for length, pressure in zip(np.concatenate((diff_y_top, diff_y_bottom)),np.concatenate((avg_pressure_top,avg_pressure_bottom))):
        force+=length*pressure  
    return force

def drag_velocity(u_inf,p_inf, p_y,rho, y_locations,u):
    diff_y = np.array([y_locations[i] - y_locations[i-1] for i in range(1, len(y_locations))])
    avg_p=p_inf-np.array([(p_y[i] + p_y[i-1])/2 for i in range(1, len(p_y))])

    drag=sum(diff_y*avg_p)

    dy=0.0001
    y=y_locations[0]
    u_avg=0
    u2_avg=0
    while y<y_locations[-1]:
        u_avg+=(u(y))*dy
        u2_avg+=(u(y))**2*dy
        y+=dy
    u_avg/=(y_locations[-1]-y_locations[0])
    u2_avg/=(y_locations[-1]-y_locations[0])
    drag+=rho*(u_inf*u_avg-u2_avg)*(y_locations[-1]-y_locations[0])
    return drag 

def lift_drag_surface_alpha(alpha,location_x_top, location_x_bottom,location_y_top,location_y_bottom,pressure_top, pessure_bottom):
    lift=normal_force(location_x_top, location_x_bottom, pressure_top, pessure_bottom)*cos(np.radians(alpha))-tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom)*sin(np.radians(alpha))
    drag=normal_force(location_x_top, location_x_bottom, pressure_top, pessure_bottom)*sin(np.radians(alpha))+tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom)*cos(np.radians(alpha))
    return np.array([lift, drag])

def lift_drag_wake_alpha(alpha,location_x_top, location_x_bottom, pressure_top, pessure_bottom,u_inf,p_inf,p_y,y_locations,rho,u,c):
    normal=normal_force(location_x_top, location_x_bottom, pressure_top, pessure_bottom)
    drag=drag_velocity(u_inf,p_inf,p_y,rho,y_locations,u)/c
    tangential=(drag-normal*sin(np.radians(alpha)))/cos(np.radians(alpha))
    lift=normal*cos(np.radians(alpha))-tangential*sin(np.radians(alpha))
    return np.array([lift, drag])

def moment(location_x_top, location_x_bottom,location_y_top,location_y_bottom, pressure_top, pessure_bottom):
    diff_top = np.array([location_x_top[i] - location_x_top[i-1] for i in range(1, len(location_x_top))])
    diff_bottom = np.array([location_x_bottom[i] - location_x_bottom[i-1] for i in range(1, len(location_x_bottom))])
    diff_y_top = np.array([location_y_top[i] - location_y_top[i-1] for i in range(1, len(location_y_top))])
    diff_y_bottom= np.array([-location_y_bottom[i] + location_y_bottom[i-1] for i in range(1, len(location_y_bottom))])

    avg_x_top= np.array([(location_x_top[i] + location_x_top[i-1])/2 for i in range(1, len(location_x_top))])
    avg_x_bottom = np.array([(location_x_bottom[i] + location_x_bottom[i-1])/2 for i in range(1, len(location_x_bottom))])
    avg_x_y_top = np.array([(location_y_top[i] + location_y_top[i-1])/2 for i in range(1, len(location_y_top))])
    avg_x_y_bottom= np.array([(location_y_bottom[i] + location_y_bottom[i-1])/2 for i in range(1, len(location_y_bottom))])

    avg_pressure_top=np.array([(pressure_top[i] + pressure_top[i-1])/2 for i in range(1, len(pressure_top))])
    avg_pressure_bottom=np.array([(pessure_bottom[i] + pessure_bottom[i-1])/2 for i in range(1, len(pessure_bottom))])

    M=0
    for length, location, pressure  in zip(diff_top,avg_x_top,avg_pressure_top):
        M+=length*pressure*location  
    for length, location, pressure  in zip(diff_bottom,avg_x_bottom,avg_pressure_bottom):
        M-=length*pressure*location  
    for length, location, pressure  in zip(diff_y_top,avg_x_y_top,avg_pressure_top):
        M+=length*pressure*location  
    for length, location, pressure  in zip(diff_y_bottom,avg_x_y_bottom,avg_pressure_bottom):
        M+=length*pressure*location  
    return M

def u_profile(rho, static_pos, static_p, total_pos, total_p,y):
    return ((np.interp(y, total_pos, total_p)-np.interp(y, static_pos, static_p))*2/rho)**(1/2)

def plot_u(u,min_y, max_y, step=0.1):
    y =np.arange(min_y, max_y + step, step)
    u_val = np.array([u(yi/1000) for yi in y])

    plt.figure(figsize=(10, 6))

    plt.plot(y, u_val, label='Wake rake velocicy profile', linewidth=2, color='tab:orange')

    plt.axhline(0, color='gray', linewidth=2, linestyle='--')
    
    plt.xlim(min_y,max_y)
    plt.ylim(0)
    plt.xlabel('y [mm]', fontsize=14, fontweight='bold')
    plt.ylabel('Velocity [m/s]', fontsize=14, fontweight='bold')
    plt.title('Wake rake velocicy vs y', fontsize=16, fontweight='bold')
    
    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()

def plot_lift(alpha_saved,lift_surface_saved,lift_wake_saved):
    n_alpha_max=np.argmax(alpha_saved)
    plt.figure(figsize=(10, 6))

    plt.plot(alpha_saved[n_alpha_max-1:], lift_surface_saved[n_alpha_max-1:], label=r'$C_l$ from airfoil pressure data, hysteresis', linewidth=linewidth_minor, color='tab:red')
    plt.plot(alpha_saved[n_alpha_max-1:], lift_wake_saved[n_alpha_max-1:], label=r'$C_l$ from airfoil pressure and wake rake data, hysteresis', linewidth=linewidth_minor, color='tab:purple')
    plt.plot(alpha_saved[0:n_alpha_max], lift_surface_saved[0:n_alpha_max], label=r'$C_l$ from airfoil pressure data', linewidth=linewidth_major, color='tab:orange')
    plt.plot(alpha_saved[0:n_alpha_max], lift_wake_saved[0:n_alpha_max], label=r'$C_l$ from airfoil pressure and wake rake data', linewidth=linewidth_major, color='tab:blue')

    scatter_point(0, lift_wake_saved[5])  # mark the point
    plt.annotate(fr"$C_{{l,max}}$={lift_wake_saved[5]:.3f}",(0, lift_wake_saved[5]),textcoords="offset points",xytext=(-100, 10),arrowprops=dict(arrowstyle="->"),fontsize=15)

    idx = np.argmax(lift_wake_saved)

    scatter_point(alpha_saved[idx], lift_wake_saved[idx])
    plt.annotate(
        fr"$C_{{l,max}}$={lift_wake_saved[idx]:.3f}",
        (alpha_saved[idx], lift_wake_saved[idx]),
        textcoords="offset points",
        xytext=(-50, -70),
        arrowprops=dict(arrowstyle="->"),
        fontsize=15
    )

    plt.axhline(0, color='gray', linewidth=2, linestyle='--')
    plt.axvline(0, color='gray', linewidth=2, linestyle='--')

    plt.xlim(min(alpha_saved),max(alpha_saved))
    plt.xlabel(r'alpha [°]', fontsize=14, fontweight='bold')
    plt.ylabel(r'$C_l$ [-]', fontsize=14, fontweight='bold')
    plt.title(r'Lift coefficient vs alpha', fontsize=16, fontweight='bold')

    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    
def plot_drag(alpha_saved,drag_surface_saved,drag_wake_saved):
    n_alpha_max=np.argmax(alpha_saved)
    plt.figure(figsize=(10, 6))

    plt.plot(alpha_saved[n_alpha_max-1:], drag_surface_saved[n_alpha_max-1:], label=r'$C_d$ from airfoil pressure data, hysteresis', linewidth=linewidth_minor, color='tab:red')
    plt.plot(alpha_saved[n_alpha_max-1:], drag_wake_saved[n_alpha_max-1:], label=r'$C_d$ from wake rake data, hysteresis', linewidth=linewidth_minor, color='tab:purple')
    plt.plot(alpha_saved[0:n_alpha_max], drag_surface_saved[0:n_alpha_max], label=r'$C_d$ from airfoil pressure data', linewidth=linewidth_major, color='tab:orange')
    plt.plot(alpha_saved[0:n_alpha_max], drag_wake_saved[0:n_alpha_max], label=r'$C_d$ from wake rake data', linewidth=linewidth_major, color='tab:blue')


    idx = np.argmin(drag_surface_saved[0:n_alpha_max])

    scatter_point(alpha_saved[idx], drag_surface_saved[idx])
    plt.annotate(
        fr"$C_{{d,min}}$={drag_surface_saved[idx]:.4f}",
        (alpha_saved[idx], drag_surface_saved[idx]),
        textcoords="offset points",
        xytext=(-120, 35),
        arrowprops=dict(arrowstyle="->"),
        fontsize=15
    )

    idx = np.argmin(drag_wake_saved[0:n_alpha_max])

    scatter_point(alpha_saved[idx], drag_wake_saved[idx])
    plt.annotate(
        fr"$C_{{d,min}}$={drag_wake_saved[idx]:.4f}",
        (alpha_saved[idx], drag_wake_saved[idx]),
        textcoords="offset points",
        xytext=(-50, 35),
        arrowprops=dict(arrowstyle="->"),
        fontsize=15
    )

    plt.axhline(0, color='gray', linewidth=2, linestyle='--')
    plt.axvline(0, color='gray', linewidth=2, linestyle='--')

    plt.xlim(min(alpha_saved),max(alpha_saved))
    plt.xlabel('alpha [°]', fontsize=14, fontweight='bold')
    plt.ylabel(r'$C_d$ [-]', fontsize=14, fontweight='bold')
    plt.title('Drag coefficient vs alpha', fontsize=16, fontweight='bold')

    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.grid(True, alpha=0.5)

    plt.tight_layout()

def plot_moment(alpha_saved,moment_surface_saved):
    n_alpha_max=np.argmax(alpha_saved)
    plt.figure(figsize=(10, 6))

    plt.plot(alpha_saved[n_alpha_max-1:], moment_surface_saved[n_alpha_max-1:], label=r'$C_m$ from airfoil pressure data, hysteresis', linewidth=linewidth_minor, color='tab:red')
    plt.plot(alpha_saved[0:n_alpha_max], moment_surface_saved[0:n_alpha_max], label=r'$C_m$ from airfoil pressure data', linewidth=linewidth_major, color='tab:orange')


    plt.axhline(0, color='gray', linewidth=2, linestyle='--')
    plt.axvline(0, color='gray', linewidth=2, linestyle='--')

    plt.xlim(min(alpha_saved),max(alpha_saved))
    plt.xlabel('alpha [°]', fontsize=14, fontweight='bold')
    plt.ylabel(r'$C_m$ [-]', fontsize=14, fontweight='bold')
    plt.title('Moment coefficient about quater chord vs alpha', fontsize=16, fontweight='bold')

    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.grid(True, alpha=0.5)

    plt.tight_layout()