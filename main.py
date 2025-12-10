import csv
import numpy as np
from numpy import cos, sin
import reference_condtions as rc


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
def tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom):
    diff_y_top = np.array([location_y_top[i] - location_y_top[i-1] for i in range(1, len(location_y_top))])
    diff_y_bottom= np.array([location_y_bottom[i] - location_y_bottom[i-1] for i in range(1, len(location_y_bottom))])
    force=0
    for length, pressure in zip(np.concatenate((diff_top, diff_bottom)),np.concatenate((avg_pressure_top,avg_pressure_bottom))):
        force-=length*pressure*c
    return force
def drag_velocity(u_inf,u_y,p_inf, p_y,rho, y_locations):
    diff_y = np.array([y_locations[i] - y_locations[i-1] for i in range(1, len(y_locations))])
    avg_u=np.array([(u_y[i] + u_y[i-1])/2 for i in range(1, len(u_y))])
    avg_p=np.array([(p_y[i] + p_y[i-1])/2 for i in range(1, len(p_y))])
    drag=0
    for length,u,p in zip(diff_y,avg_u,avg_p):
        drag+=((rho*(u_inf-u)*u)+(p_inf-p))*length
    return drag

def lift_drag_surface_alpha(alpha):
    lift=normal_force(c,location_top, location_bottom, pressure_top, pessure_bottom)*cos(alpha)-tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom)*sin(alpha)
    drag=normal_force(c,location_top, location_bottom, pressure_top, pessure_bottom)*sin(alpha)+tangential_force(location_y_top,location_y_bottom, pressure_top, pessure_bottom)*cos(alpha)
    return lift, drag

lift_drag_wake_alpha(alpha):
    normal=normal_force(c,location_top, location_bottom, pressure_top, pessure_bottom)
    tangential=(drag_velocity(u_inf,u_y,p_inf, p_y,rho, y_locations)-norma*sin(alpha))/cos(alpha)
    lift=normal*cos(alpha)-tangential*sin(alpha)
    drag=+normal*sin(alpha)+tangentiatangential*cos(alpha)
    return lift,drag
    
    


filename = "raw_Group12_2D.csv"

run_number = []
time = []
alpha = []
delta_pb_value = []
p_bar_value = []
temperature = []
rpm = []
density = []
pressure_taps = []

with open(filename, "r", encoding="utf-8") as data_file:
    reader = csv.reader(data_file, skipinitialspace=True)
    header = next(reader)
    next(reader, None)
    p_indices = [i for i, name in enumerate(header) if name.strip().startswith("P")]

    for row in reader:
        if not row or not row[0].strip():
            continue

        row = [cell.strip() for cell in row]

        run_number.append(int(row[0]))
        time.append(row[1])
        alpha.append(float(row[2]))
        delta_pb_value.append(float(row[3]))
        p_bar_value.append(float(row[4]))
        temperature.append(float(row[5]))
        rpm.append(float(row[6]))
        density.append(float(row[7]))

        tap_values = [float(row[i]) for i in p_indices]
        pressure_taps.append(tap_values)

ndata = len(run_number)

avg_temp = sum(temperature) / ndata
avg_pbar = sum(p_bar_value) / ndata
avg_dpb = sum(delta_pb_value) / ndata

dynamic_pressure = rc.calculate_reference_dynamic_pressure(avg_dpb)

print("air density:", rc.calculate_air_density(avg_pbar, avg_temp))
print("dynamic viscosity:", rc.calculate_dynamic_viscosity(avg_temp))
print("reference dynamic pressure:", dynamic_pressure)
print("reference static pressure:", rc.calculate_reference_static_pressure(avg_pbar, dynamic_pressure))

