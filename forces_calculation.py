import numpy as np
from numpy import cos, sin
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.rcParams['mathtext.default'] = 'regular'
linewidth_major=2
linewidth_minor=1
marker_s_main=5
marker_s_minor=3
axis_font=20
anotation_font=20
legend_font=16
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

def plot_u(u,min_y, max_y, step=0,u_inf=20):
    y =np.arange(min_y, max_y + step, step)
    u_val = np.array([u(yi/1000) for yi in y])

    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(y, u_val, label='Wake rake velocicy profile', linewidth=2, color='tab:orange')
    
    scatter_point(min_y, u(min_y),s=100)
    plt.annotate(
        fr"{u(min_y):.3f}",
        (min_y, u(min_y)),
        textcoords="offset points",
        xytext=(18, -70),
        arrowprops=dict(arrowstyle="->"),
        fontsize=anotation_font,
        color = "#1f4ed8"
    )

    scatter_point(max_y, u(max_y),s=100)
    plt.annotate(
        fr"{u(max_y):.3f}",
        (max_y, u(max_y)),
        textcoords="offset points",
        xytext=(-71, -70),
        arrowprops=dict(arrowstyle="->"),
        fontsize=anotation_font,
        color = "#1f4ed8"
    )

    plt.text(
    0.5, 0.2,
    f"$U_{{\infty}}$={u_inf:.3f} m/s",
    transform=plt.gca().transAxes,
    ha="center",
    va="top",
    fontsize=anotation_font,
    color = "#1f4ed8"
    )
    
    plt.xlim(min_y,max_y)
    plt.ylim(0)
    plt.xlabel(r'Wake Rake Location [mm]',fontsize=axis_font)
    plt.ylabel(r'Velocity (u) [m/s]',fontsize=axis_font)
    
    
    plt.legend(loc='lower right',fontsize=legend_font)

def write_vaues(alpha_saved,lift_wake_saved,drag_wake_saved, moment_surface_saved):
    n_alpha_max=np.argmax(alpha_saved)+1
    import csv

    with open('data files/forces_files/forces.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        # optional header
        writer.writerow(['alpha', 'cl', 'cd', 'cm'])

        for alpha, cl, cd, cm in zip(
            alpha_saved[:n_alpha_max],
            lift_wake_saved[:n_alpha_max],
            drag_wake_saved[:n_alpha_max],
            moment_surface_saved[:n_alpha_max]
        ):
            writer.writerow([
                f"{alpha:.6f}",
                f"{cl:.6f}",
                f"{cd:.6f}",
                f"{cm:.6f}"])




def plot_lift(alpha_saved,lift_surface_saved,lift_wake_saved):
    alpha_saved=np.array(alpha_saved)
    lift_wake_saved=np.array(lift_wake_saved)
    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(alpha_saved[n_alpha_max-1:], lift_surface_saved[n_alpha_max-1:], label=r'Airfoil pressure data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[n_alpha_max-1:], lift_wake_saved[n_alpha_max-1:], label=r'Airfoil pressure and wake rake data, hysteresis', linewidth=linewidth_minor, color='tab:purple',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[0:n_alpha_max], lift_surface_saved[0:n_alpha_max], label=r'Airfoil pressure data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)
    plt.plot(alpha_saved[0:n_alpha_max], lift_wake_saved[0:n_alpha_max], label=r'Airfoil pressure and wake rake data', linewidth=linewidth_major, color='tab:blue',marker='o',markersize=marker_s_main)

    scatter_point(0, lift_wake_saved[5])
    plt.annotate(fr"$C_{{l,0}}$={lift_wake_saved[5]:.3f}",(0, lift_wake_saved[5]),textcoords="offset points",xytext=(60, -5),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "#1f4ed8")

    idx = np.argmax(lift_wake_saved)

    scatter_point(alpha_saved[idx], lift_wake_saved[idx])
    plt.annotate(
        f"$\\alpha$={alpha_saved[idx]}\n$C_{{l,max}}$={lift_wake_saved[idx]:.3f}",
        (alpha_saved[idx], lift_wake_saved[idx]),
        textcoords="offset points",
        xytext=(-50, -107),
        arrowprops=dict(arrowstyle="->"),
        fontsize=anotation_font,
        color = "#1f4ed8"
    )

    mask =(alpha_saved <= 9)
    alpha_lin = alpha_saved[mask]
    cl_lin = lift_wake_saved[mask]

    slope, intercept = np.polyfit(alpha_lin, cl_lin, 1)

    alpha_fit = np.linspace(alpha_lin.min(), alpha_lin.max(), 100)
    cl_fit = slope * alpha_fit + intercept

    plt.plot(alpha_fit, cl_fit, 'k--', linewidth=2)

    plt.annotate(
    fr"$\frac{{dC_l}}{{d\alpha}}={slope:.3f}\ \mathrm{{deg}}^{{-1}}$",
    xy=(alpha_fit.mean(), cl_fit.mean()),
    textcoords="offset points",
    xytext=(-78, 95),
    arrowprops=dict(arrowstyle="->"),
    fontsize=anotation_font,
    color = "#1f4ed8")

    plt.xlim(min(alpha_saved),max(alpha_saved))
    plt.xlabel(r"Angle of Attack $\alpha$ [deg]",fontsize=axis_font)
    plt.ylabel(r"Lift Coefficient ($C_l$) [-]",fontsize=axis_font)
    
    plt.legend(loc='lower right',fontsize=legend_font)
 
def plot_drag(alpha_saved,drag_surface_saved,drag_wake_saved):
    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(alpha_saved[n_alpha_max-1:], drag_surface_saved[n_alpha_max-1:], label=r'Airfoil pressure data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[n_alpha_max-1:], drag_wake_saved[n_alpha_max-1:], label=r'Wake rake data, hysteresis', linewidth=linewidth_minor, color='tab:purple',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[0:n_alpha_max], drag_surface_saved[0:n_alpha_max], label=r'Airfoil pressure data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)
    plt.plot(alpha_saved[0:n_alpha_max], drag_wake_saved[0:n_alpha_max], label=r'Wake rake data', linewidth=linewidth_major, color='tab:blue',marker='o',markersize=marker_s_main)


    idx = np.argmin(drag_surface_saved[0:n_alpha_max])

    scatter_point(alpha_saved[idx], drag_surface_saved[idx])
    plt.annotate(
        f"$\\alpha$={alpha_saved[idx]}\n$C_{{d,min}}$={drag_surface_saved[idx]:.4f}",
        (alpha_saved[idx], drag_surface_saved[idx]),
        textcoords="offset points",
        xytext=(7, 50),
        arrowprops=dict(arrowstyle="->"),
        fontsize=anotation_font,
        color = "tab:orange"
    )

    idx = np.argmin(drag_wake_saved[0:n_alpha_max])

    scatter_point(alpha_saved[idx], drag_wake_saved[idx])
    plt.annotate(
        f"$\\alpha$={alpha_saved[idx]}\n$C_{{d,min}}$={drag_wake_saved[idx]:.4f}",
        (alpha_saved[idx], drag_wake_saved[idx]),
        textcoords="offset points",
        xytext=(0, 50),
        arrowprops=dict(arrowstyle="->"),
        fontsize=anotation_font,
        color = "#1f4ed8"
    )

    plt.xlim(min(alpha_saved),max(alpha_saved))
    plt.xlabel(r"Angle of Attack $\alpha$ [deg]",fontsize=axis_font)
    plt.ylabel(r'Drag Coefficient ($C_d$) [-]',fontsize=axis_font)

    plt.legend(fontsize=legend_font)

def plot_moment(alpha_saved,moment_surface_saved):
    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(alpha_saved[n_alpha_max-1:], moment_surface_saved[n_alpha_max-1:], label=r'Airfoil pressure data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[0:n_alpha_max], moment_surface_saved[0:n_alpha_max], label=r'Airfoil pressure data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)

    plt.xlabel(r"Angle of Attack $\alpha$ [deg]",fontsize=axis_font)
    plt.ylabel(r'Moment Coefficient ($C_{m,0.25}$) [-]',fontsize=axis_font)

    
    plt.legend(fontsize=legend_font)

def plot_cl_cd(alpha_saved,lift_surface_saved,lift_wake_saved,drag_surface_saved,drag_wake_saved):
    lift_surface_saved=np.array(lift_surface_saved)
    lift_wake_saved=np.array(lift_wake_saved)
    drag_surface_saved=np.array(drag_surface_saved)
    drag_wake_saved=np.array(drag_wake_saved)

    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(drag_surface_saved[n_alpha_max-1:], lift_surface_saved[n_alpha_max-1:], label=r'Airfoil pressure data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(drag_wake_saved[n_alpha_max-1:], lift_wake_saved[n_alpha_max-1:], label=r'Wake rake data, hysteresis', linewidth=linewidth_minor, color='tab:purple',marker='^',markersize=marker_s_minor)

    plt.plot(drag_surface_saved[0:n_alpha_max], lift_surface_saved[0:n_alpha_max], label=r'Airfoil pressure data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)
    plt.plot(drag_wake_saved[0:n_alpha_max], lift_wake_saved[0:n_alpha_max], label=r'Wake rake data', linewidth=linewidth_major, color='tab:blue',marker='o',markersize=marker_s_main)
    
    divide=lift_surface_saved[0:n_alpha_max]//drag_surface_saved[0:n_alpha_max]
    idx = np.argmax(divide)
    scatter_point(drag_surface_saved[idx], lift_surface_saved[idx])
    plt.annotate(f"$\\alpha$={alpha_saved[idx]}\n$(C_l/C_d)_{{max}}$={divide[idx]:.2f}",(drag_surface_saved[idx], lift_surface_saved[idx]),textcoords="offset points",xytext=(70, -15),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "tab:orange")

    divide=lift_wake_saved[0:n_alpha_max]//drag_wake_saved[0:n_alpha_max]
    idx = np.argmax(divide)
    scatter_point(drag_wake_saved[idx], lift_wake_saved[idx])
    plt.annotate(f"$\\alpha$={alpha_saved[idx]}\n$(C_l/C_d)_{{max}}$={divide[idx]:.2f}",(drag_wake_saved[idx], lift_wake_saved[idx]),textcoords="offset points",xytext=(80, -40),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "#1f4ed8")


    plt.xlim(0)
    plt.xlabel(r'Drag Coefficient ($C_d$) [-]',fontsize=axis_font)
    plt.ylabel(r'Lift Coefficient  ($C_l$)  [-]',fontsize=axis_font)

    
    plt.legend(loc='lower right',fontsize=legend_font)

def plot_k(alpha_saved,k_saved):
    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(alpha_saved[n_alpha_max-1:], k_saved[n_alpha_max-1:], label=r'Center of pressure location, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[0:n_alpha_max], k_saved[0:n_alpha_max], label=r'Center of pressure location', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)

    plt.xlabel(r"Angle of Attack $\alpha$ [deg]",fontsize=axis_font)
    plt.ylabel(r'Fraction of Chord (k) [-]',fontsize=axis_font)

    plt.xlim(min(alpha_saved),max(alpha_saved))
    # plt.ylim(0,1)

    
    plt.legend(fontsize=legend_font)

def plot_cl_corr(alpha_saved,lift_wake_saved):
    import csv

    with open("data files\corrected_forces.csv", newline="") as f:
        reader = csv.DictReader(f)
        
        alpha_new = []
        lift_new = []

        for row in reader:
            alpha_new.append(float(row["alpha"]))
            lift_new.append(float(row["cl"]))
    n_alpha_max=np.argmax(alpha_saved)+1
    alpha_new=np.array(alpha_new)
    lift_new=np.array(lift_new)
    alpha_saved=np.array(alpha_saved)
    lift_wake_saved=np.array(lift_wake_saved)

    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(alpha_new[n_alpha_max-1:], lift_new[n_alpha_max-1:], label=r'Corrected data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[n_alpha_max-1:], lift_wake_saved[n_alpha_max-1:], label=r'Uncorrected data, hysteresis', linewidth=linewidth_minor, color='tab:purple',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_new[0:n_alpha_max], lift_new[0:n_alpha_max], label=r'Corrected data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)
    plt.plot(alpha_saved[0:n_alpha_max], lift_wake_saved[0:n_alpha_max], label=r'Uncorrected data', linewidth=linewidth_major, color='tab:blue',marker='o',markersize=marker_s_main)

    scatter_point(0, lift_wake_saved[5])
    plt.annotate(fr"$C_{{l,0}}$={lift_wake_saved[5]:.3f}",(0, lift_wake_saved[5]),textcoords="offset points",xytext=(45, -5),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "#1f4ed8")

    idx = np.argmax(lift_wake_saved)

    scatter_point(alpha_saved[idx], lift_wake_saved[idx])
    plt.annotate(f"$\\alpha$={alpha_saved[idx]}\n$C_{{l,max}}$={lift_wake_saved[idx]:.3f}",(alpha_saved[idx], lift_wake_saved[idx]),textcoords="offset points",xytext=(-100, -140),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "#1f4ed8")

    idx = np.argmax(lift_new)

    scatter_point(alpha_new[idx], lift_new[idx])
    plt.annotate(f"$\\alpha$={alpha_new[idx]}\n$C_{{l,max}}$={lift_new[idx]:.3f}",(alpha_new[idx], lift_new[idx]),textcoords="offset points",xytext=(20, -120),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "tab:orange")

    mask =(alpha_new <= 9)
    alpha_lin = alpha_new[mask]
    cl_lin = lift_new[mask]

    slope, intercept = np.polyfit(alpha_lin, cl_lin, 1)

    alpha_fit = np.linspace(alpha_lin.min(), alpha_lin.max(), 100)
    cl_fit = slope * alpha_fit + intercept

    plt.plot(alpha_fit, cl_fit, 'k--', linewidth=2)

    plt.annotate(
    fr"$\frac{{dC_l}}{{d\alpha}}={slope:.3f}\ \mathrm{{deg}}^{{-1}}$",
    xy=(alpha_fit.mean(), cl_fit.mean()),
    textcoords="offset points",
    xytext=(70, -26),
    arrowprops=dict(arrowstyle="->"),
    fontsize=anotation_font,
    color = "tab:orange")

    mask =(alpha_saved <= 9)
    alpha_lin = alpha_saved[mask]
    cl_lin = lift_wake_saved[mask]

    slope, intercept = np.polyfit(alpha_lin, cl_lin, 1)

    alpha_fit = np.linspace(alpha_lin.min(), alpha_lin.max(), 100)
    cl_fit = slope * alpha_fit + intercept

    plt.plot(alpha_fit, cl_fit, 'k--', linewidth=2)

    plt.annotate(
    fr"$\frac{{dC_l}}{{d\alpha}}={slope:.3f}\ \mathrm{{deg}}^{{-1}}$",
    xy=(alpha_fit.mean(), cl_fit.mean()),
    textcoords="offset points",
    xytext=(-83, 100),
    arrowprops=dict(arrowstyle="->"),
    fontsize=anotation_font,
    color = "#1f4ed8")



    plt.xlim(min(alpha_saved),max(alpha_saved))
    plt.xlabel(r"Angle of Attack $\alpha$ [deg]",fontsize=axis_font)
    plt.ylabel(r"Lift Coefficient ($C_l$) [-]",fontsize=axis_font)
    
    plt.legend(loc='lower right',fontsize=legend_font)

def plot_drag_corr(alpha_saved,drag_wake_saved):
    import csv

    with open("data files\corrected_forces.csv", newline="") as f:
        reader = csv.DictReader(f)
        
        alpha_new = []
        drag_new = []

        for row in reader:
            alpha_new.append(float(row["alpha"]))
            drag_new.append(float(row["cd"]))
    alpha_new=np.array(alpha_new)
    drag_new=np.array(drag_new)
    alpha_saved=np.array(alpha_saved)
    drag_wake_saved=np.array(drag_wake_saved)

    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(alpha_saved[n_alpha_max-1:], drag_new[n_alpha_max-1:], label=r'Corrected data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[n_alpha_max-1:], drag_wake_saved[n_alpha_max-1:], label=r'Uncorrected data, hysteresis', linewidth=linewidth_minor, color='tab:purple',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[0:n_alpha_max], drag_new[0:n_alpha_max], label=r'Corrected data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)
    plt.plot(alpha_saved[0:n_alpha_max], drag_wake_saved[0:n_alpha_max], label=r'Uncorrected data', linewidth=linewidth_major, color='tab:blue',marker='o',markersize=marker_s_main)


    idx = np.argmin(drag_new[0:n_alpha_max])

    scatter_point(alpha_new[idx], drag_new[idx])
    plt.annotate(
        f"$\\alpha$={alpha_new[idx]}\n$C_{{d,min}}$={drag_new[idx]:.4f}",
        (alpha_new[idx], drag_new[idx]),
        textcoords="offset points",
        xytext=(47, 60),
        arrowprops=dict(arrowstyle="->"),
        fontsize=anotation_font,
        color = "tab:orange"
    )

    idx = np.argmin(drag_wake_saved[0:n_alpha_max])

    scatter_point(alpha_saved[idx], drag_wake_saved[idx])
    plt.annotate(
        f"$\\alpha$={alpha_saved[idx]}\n$C_{{d,min}}$={drag_wake_saved[idx]:.4f}",
        (alpha_saved[idx], drag_wake_saved[idx]),
        textcoords="offset points",
        xytext=(-120, 60),
        arrowprops=dict(arrowstyle="->"),
        fontsize=anotation_font,
        color = "#1f4ed8"
    )

    plt.xlim(min(alpha_saved),max(alpha_saved))
    plt.xlabel(r"Angle of Attack $\alpha$ [deg]",fontsize=axis_font)
    plt.ylabel(r'Drag Coefficient ($C_d$) [-]',fontsize=axis_font)

    
    plt.legend(fontsize=legend_font)

def plot_moment_corr(alpha_saved,moment_surface_saved):
    import csv

    with open("data files\corrected_forces.csv", newline="") as f:
        reader = csv.DictReader(f)
        
        alpha_new = []
        moment_new = []

        for row in reader:
            alpha_new.append(float(row["alpha"]))
            moment_new.append(float(row["cm"]))

    alpha_new=np.array(alpha_new)
    moment_new=np.array(moment_new)
    alpha_saved=np.array(alpha_saved)
    moment_surface_saved=np.array(moment_surface_saved)

    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    plt.plot(alpha_saved[n_alpha_max-1:], moment_new[n_alpha_max-1:], label=r'Corrected data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[n_alpha_max-1:], moment_surface_saved[n_alpha_max-1:], label=r'Uncorrected data, hysteresis', linewidth=linewidth_minor, color='tab:purple',marker='^',markersize=marker_s_minor)
    plt.plot(alpha_saved[0:n_alpha_max], moment_new[0:n_alpha_max], label=r'Corrected data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)
    plt.plot(alpha_saved[0:n_alpha_max], moment_surface_saved[0:n_alpha_max], label=r'Uncorrected data', linewidth=linewidth_major, color='tab:blue',marker='o',markersize=marker_s_main)

    plt.xlabel(r"Angle of Attack $\alpha$ [deg]",fontsize=axis_font)
    plt.ylabel(r'Moment Coefficient ($C_{m,0.25}$) [-]',fontsize=axis_font)

    
    plt.legend(fontsize=legend_font)

def plot_cl_cd_corr(alpha_saved,lift_wake_saved,drag_wake_saved):
    import csv

    with open("data files\corrected_forces.csv", newline="") as f:
        reader = csv.DictReader(f)
        
        alpha_new = []
        lift_new = []
        drag_new=[]

        for row in reader:
            alpha_new.append(float(row["alpha"]))
            lift_new.append(float(row["cl"]))
            drag_new.append(float(row["cd"]))
    
    alpha_new=np.array(alpha_new)
    lift_new=np.array(lift_new)
    drag_new=np.array(drag_new)
    alpha_saved=np.array(alpha_saved)
    lift_wake_saved=np.array(lift_wake_saved)
    drag_wake_saved=np.array(drag_wake_saved)

    n_alpha_max=np.argmax(alpha_saved)+1
    plt.figure(figsize=(7,5))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")
    

    plt.plot(drag_new[n_alpha_max-1:], lift_new[n_alpha_max-1:], label=r'Corrected data, hysteresis', linewidth=linewidth_minor, color='tab:red',marker='^',markersize=marker_s_minor)
    plt.plot(drag_wake_saved[n_alpha_max-1:], lift_wake_saved[n_alpha_max-1:], label=r'Uncorrected data, hysteresis', linewidth=linewidth_minor, color='tab:purple',marker='^',markersize=marker_s_minor)

    plt.plot(drag_new[0:n_alpha_max], lift_new[0:n_alpha_max], label=r'Corrected data', linewidth=linewidth_major, color='tab:orange',marker='o',markersize=marker_s_main)
    plt.plot(drag_wake_saved[0:n_alpha_max], lift_wake_saved[0:n_alpha_max], label=r'Uncorrected data', linewidth=linewidth_major, color='tab:blue',marker='o',markersize=marker_s_main)


    divide=lift_new[0:n_alpha_max]//drag_new[0:n_alpha_max]
    idx = np.argmax(divide)
    scatter_point(drag_new[idx], lift_new[idx])
    plt.annotate(f"$\\alpha$={alpha_new[idx]}\n$(C_l/C_d)_{{max}}$={divide[idx]:.2f}",(drag_new[idx], lift_new[idx]),textcoords="offset points",xytext=(60, -110),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "tab:orange")

    divide=lift_wake_saved[0:n_alpha_max]//drag_wake_saved[0:n_alpha_max]
    idx = np.argmax(divide)
    scatter_point(drag_wake_saved[idx], lift_wake_saved[idx])
    plt.annotate(f"$\\alpha$={alpha_saved[idx]}\n$(C_l/C_d)_{{max}}$={divide[idx]:.2f}",(drag_wake_saved[idx], lift_wake_saved[idx]),textcoords="offset points",xytext=(58, -35),arrowprops=dict(arrowstyle="->"),fontsize=anotation_font,color = "#1f4ed8")



    plt.xlim(0)
    plt.xlabel(r'Drag Coefficient ($C_d$) [-]',fontsize=axis_font)
    plt.ylabel(r'Lift Coefficient  ($C_l$)  [-]',fontsize=axis_font)

    
    plt.legend(loc='lower right',fontsize=legend_font)