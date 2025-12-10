import csv
from pathlib import Path

import pandas as pd
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

def tangential_force(c,location_y_top,location_y_bottom, pressure_top, pessure_bottom):
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

    
    


CSV_FILE = Path("raw_Group12_2D.csv")
EXCEL_FILE = Path("SLT practical coordinates.xlsx")

run_number = []
time = []
alpha = []
delta_pb_value = []
p_bar_value = []
temperature = []
rpm = []
density = []
pressure_taps = []

with CSV_FILE.open("r", encoding="utf-8") as data_file:
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

airfoil_pressure_tap_coordinates = []
total_wake_rake_locations = []
static_wake_rake_locations = []
pitot_static_ports = []

if EXCEL_FILE.exists():
    excel_df = pd.read_excel(EXCEL_FILE, sheet_name=0)
    excel_df.columns = [col.strip() if isinstance(col, str) else col for col in excel_df.columns]
    excel_df = excel_df.rename(
        columns={
            "airfoil pressure tap coordinates": "x_percent",
            "Unnamed: 2": "y_percent",
            "Pressure no.": "total_wake_pressure_no",
            "total wake rake probe locations [mm]": "total_wake_location_mm",
            "pressure no..1": "static_wake_pressure_no",
            "static wake rake probe locations [mm]": "static_wake_location_mm",
            "pressure no..2": "pitot_pressure_no",
            "pitot-static tube at the wall of the test section": "pitot_description",
        }
    )

    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for _, row in excel_df.dropna(subset=["pressure no."]).iterrows():
        x_val = to_float(row.get("x_percent"))
        y_val = to_float(row.get("y_percent"))
        if x_val is None or y_val is None:
            continue
        tap_name = str(row["pressure no."]).strip()
        airfoil_pressure_tap_coordinates.append((tap_name, x_val, y_val))

    for _, row in excel_df.dropna(subset=["total_wake_pressure_no"]).iterrows():
        location = to_float(row.get("total_wake_location_mm"))
        if location is None:
            continue
        probe_name = str(row["total_wake_pressure_no"]).strip()
        total_wake_rake_locations.append((probe_name, location))

    for _, row in excel_df.dropna(subset=["static_wake_pressure_no"]).iterrows():
        location = to_float(row.get("static_wake_location_mm"))
        if location is None:
            continue
        probe_name = str(row["static_wake_pressure_no"]).strip()
        static_wake_rake_locations.append((probe_name, location))

    for _, row in excel_df.dropna(subset=["pitot_pressure_no"]).iterrows():
        description = row.get("pitot_description")
        if not isinstance(description, str):
            continue
        pitot_static_ports.append((description.strip(), str(row["pitot_pressure_no"]).strip()))
else:
    print(f"Excel file not found: {EXCEL_FILE}")

print("airfoil pressure taps:", airfoil_pressure_tap_coordinates)
print("wake rake total ports:", total_wake_rake_locations)
print("wake rake static ports:", static_wake_rake_locations)
print("pitot-static ports:", pitot_static_ports)