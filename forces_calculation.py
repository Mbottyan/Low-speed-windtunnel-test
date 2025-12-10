import numpy as np
from numpy import cos, sin
import pandas as pd
def normal_force(c,location_top, location_bottom, pressure_top, pessure_bottom):
    diff_top = np.array([location_top[i] - location_top[i-1] for i in range(1, len(location_top))])
    diff_bottom = np.array([location_bottom[i] - location_bottom[i-1] for i in range(1, len(location_bottom))])
    avg_pressure_top=np.array([(pressure_top[i] + pressure_top[i-1])/2 for i in range(1, len(pressure_top))])
    avg_pressure_bottom=np.array([(pessure_bottom[i] + pessure_bottom[i-1])/2 for i in range(1, len(pessure_bottom))])
    force=0
    for length, pressure in zip(diff_top,avg_pressure_top):
        force-=length*pressure*c
    for length, pressure in zip(diff_bottom,avg_pressure_bottom):
        force+=length*pressure*c
    return force

def tangential_force(c,location_y_top,location_y_bottom, pressure_top, pessure_bottom):
    diff_y_top = np.array([location_y_top[i] - location_y_top[i-1] for i in range(1, len(location_y_top))])
    diff_y_bottom= np.array([location_y_bottom[i] - location_y_bottom[i-1] for i in range(1, len(location_y_bottom))])
    avg_pressure_top=np.array([(pressure_top[i] + pressure_top[i-1])/2 for i in range(1, len(pressure_top))])
    avg_pressure_bottom=np.array([(pessure_bottom[i] + pessure_bottom[i-1])/2 for i in range(1, len(pessure_bottom))])
    force=0
    for length, pressure in zip(np.concatenate((diff_y_top, diff_y_bottom)),np.concatenate((avg_pressure_top,avg_pressure_bottom))):
        force-=length*pressure*c
    return force
def drag_velocity(u_inf,u_y,p_inf, p_y,rho, y_locations):
    diff_y = np.array([y_locations[i] - y_locations[i-1] for i in range(1, len(y_locations))])
    avg_u=np.array([(u_y[i] + u_y[i-1])/2 for i in range(1, len(u_y))])
    avg_p=np.array([(p_y[i] + p_y[i-1])/2 for i in range(1, len(p_y))])
    drag=0
    for length,u,p in zip(diff_y,avg_u,avg_p):
        drag+=((rho*(u_inf-u)*u)+(p_inf-p))*length*c
    return drag

def lift_drag_surface_alpha(alpha):
    lift=normal_force(c,location_top, location_bottom, pressure_top, pessure_bottom)*cos(alpha)-tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom)*sin(alpha)
    drag=normal_force(c,location_top, location_bottom, pressure_top, pessure_bottom)*sin(alpha)+tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom)*cos(alpha)
    return lift, drag

def lift_drag_wake_alpha(alpha):
    normal=normal_force(c,location_top, location_bottom, pressure_top, pessure_bottom)
    tangential=(drag_velocity(u_inf,u_y,p_inf, p_y,rho, y_locations)-normal*sin(alpha))/cos(alpha)
    lift=normal*cos(alpha)-tangential*sin(alpha)
    drag=+normal*sin(alpha)+tangential*cos(alpha)
    return lift,drag

def moment(c,location_top, location_bottom,location_y_top,location_y_bottom, pressure_top, pessure_bottom):
    diff_top = np.array([location_top[i] - location_top[i-1] for i in range(1, len(location_top))])
    diff_bottom = np.array([location_bottom[i] - location_bottom[i-1] for i in range(1, len(location_bottom))])
    diff_y_top = np.array([location_y_top[i] - location_y_top[i-1] for i in range(1, len(location_y_top))])
    diff_y_bottom= np.array([location_y_bottom[i] - location_y_bottom[i-1] for i in range(1, len(location_y_bottom))])

    avg_x_top= np.array([(location_top[i] + location_top[i-1])/2 for i in range(1, len(location_top))])
    avg_x_bottom = np.array([(location_bottom[i] + location_bottom[i-1])/2 for i in range(1, len(location_bottom))])
    avg_x_y_top = np.array([(location_y_top[i] + location_y_top[i-1])/2 for i in range(1, len(location_y_top))])
    avg_x_y_bottom= np.array([(location_y_bottom[i] + location_y_bottom[i-1])/2 for i in range(1, len(location_y_bottom))])

    avg_pressure_top=np.array([(pressure_top[i] + pressure_top[i-1])/2 for i in range(1, len(pressure_top))])
    avg_pressure_bottom=np.array([(pessure_bottom[i] + pessure_bottom[i-1])/2 for i in range(1, len(pessure_bottom))])

    M=0
    for length, location, pressure  in zip(diff_top,avg_x_top,avg_pressure_top):
        M+=length*pressure*location*c
    for length, location, pressure  in zip(diff_bottom,avg_x_bottom,avg_pressure_bottom):
        M-=length*pressure*location*c
    for length, location, pressure  in zip(diff_y_top,avg_x_y_top,avg_pressure_top):
        M+=length*pressure*location*c
    for length, location, pressure  in zip(diff_y_bottom,avg_x_y_bottom,avg_pressure_bottom):
        M-=length*pressure*location*c
    return M
