import csv
from pathlib import Path

import pandas as pd
import numpy as np
from numpy import cos, sin #is this needed?
import reference_condtions as rc
import pressure_coefficient as pc
import forces_calculation as fc
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.default'] = 'regular'



def plot_forces():
    alpha_saved=[]
    lift_surface_saved=[]
    drag_surface_saved=[]
    moment_surface_saved=[]
    lift_wake_saved=[]
    drag_wake_saved=[]
    
    for measurment_n in run_number:
        alpha=alpha_array[measurment_n-1]
        p_atm=p_bar_value[measurment_n-1]
        p_total_pivot=total_pressure_pivot[measurment_n-1]
        q_inf=rc.calculate_reference_dynamic_pressure(delta_pb_value[measurment_n-1])
        p_inf=rc.calculate_reference_static_pressure(p_total_pivot,q_inf)
        rho=rc.calculate_air_density(p_atm,temperature[measurment_n-1])
        u_inf=rc.calculate_reference_velocity(rho, q_inf)

        wake_u=lambda y: fc.u_profile(rho,static_wake_rake_locations,static_pressure_taps_wake[measurment_n-1],total_wake_rake_locations,total_pressure_taps_wake[measurment_n-1],y)

        alpha_saved.append(alpha)
        lift,drag=fc.lift_drag_surface_alpha(alpha,airfoil_taps_location_top[:, 0],airfoil_taps_location_bottom[:, 0],airfoil_taps_location_top[:, 1],airfoil_taps_location_bottom[:, 1],airfoil_taps_pressure_top[measurment_n-1],airfoil_taps_pressure_bottom[measurment_n-1])/q_inf
        moment=fc.moment(airfoil_taps_location_top[:, 0],airfoil_taps_location_bottom[:, 0],airfoil_taps_location_top[:, 1],airfoil_taps_location_bottom[:, 1],airfoil_taps_pressure_top[measurment_n-1],airfoil_taps_pressure_bottom[measurment_n-1])/q_inf
        
        lift_surface_saved.append(lift)
        drag_surface_saved.append(drag)
        moment_surface_saved.append(moment+0.25*fc.normal_force(airfoil_taps_location_top[:, 0],airfoil_taps_location_bottom[:, 0],airfoil_taps_pressure_top[measurment_n-1],airfoil_taps_pressure_bottom[measurment_n-1])/q_inf)

        lift,drag=fc.lift_drag_wake_alpha(alpha,airfoil_taps_location_top[:, 0],airfoil_taps_location_bottom[:, 0],airfoil_taps_pressure_top[measurment_n-1],airfoil_taps_pressure_bottom[measurment_n-1],u_inf,p_inf,static_pressure_taps_wake[measurment_n-1],static_wake_rake_locations,rho,wake_u,c)/q_inf
        
        lift_wake_saved.append(lift)
        drag_wake_saved.append(drag)

    alpha_saved=np.array(alpha_saved)
    lift_surface_saved=np.array(lift_surface_saved)
    drag_surface_saved=np.array(drag_surface_saved)
    moment_surface_saved=np.array(moment_surface_saved)
    lift_wake_saved=np.array(lift_wake_saved)
    drag_wake_saved=np.array(drag_wake_saved)

    # fc.plot_u(wake_u,0,220,0.01)
    fc.plot_lift(alpha_saved,lift_surface_saved,lift_wake_saved)
    fc.plot_drag(alpha_saved,drag_surface_saved,drag_wake_saved)
    fc.plot_moment(alpha_saved,moment_surface_saved)
    plt.show()

c=0.16
CSV_FILE = Path("data files/raw_Group12_2D.csv")
EXCEL_FILE = Path("data files/SLT practical coordinates.xlsx")

run_number = []
time = []
alpha_array = []
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
    p_indices = [i for i, name in enumerate(header) if name.strip().startswith("P")][1:]

    for row in reader:
        if not row or not row[0].strip():
            continue

        row = [cell.strip() for cell in row]

        run_number.append(int(row[0]))
        time.append(row[1])
        alpha_array.append(float(row[2]))
        delta_pb_value.append(float(row[3]))
        p_bar_value.append(float(row[4])*100)
        temperature.append(float(row[5]))
        rpm.append(float(row[6]))
        density.append(float(row[7]))

        tap_values = [float(row[i]) for i in p_indices]
        pressure_taps.append(tap_values)

airfoil_taps_pressure_top=[[(j+p_bar) for j in i[:25]] for i,p_bar in zip(pressure_taps,p_bar_value)]
airfoil_taps_pressure_bottom=[[(j+p_bar) for j in i[25:49]] for i,p_bar in zip(pressure_taps,p_bar_value)]

total_pressure_pivot=[(i[96]+p_bar) for i,p_bar in zip(pressure_taps,p_bar_value)]

total_pressure_taps_wake=[[(j+p_bar) for j in i[49:96]] for i,p_bar in zip(pressure_taps,p_bar_value)]
static_pressure_taps_wake=[[(j+p_bar) for j in i[97:109]] for i,p_bar in zip(pressure_taps,p_bar_value)]


n_measurements = len(run_number)

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
        total_wake_rake_locations.append((location))

    for _, row in excel_df.dropna(subset=["static_wake_pressure_no"]).iterrows():
        location = to_float(row.get("static_wake_location_mm"))
        if location is None:
            continue
        probe_name = str(row["static_wake_pressure_no"]).strip()
        static_wake_rake_locations.append((location))

    for _, row in excel_df.dropna(subset=["pitot_pressure_no"]).iterrows():
        description = row.get("pitot_description")
        if not isinstance(description, str):
            continue
        pitot_static_ports.append((description.strip(), str(row["pitot_pressure_no"]).strip()))
else:
    print(f"Excel file not found: {EXCEL_FILE}")

static_wake_rake_locations=np.array(static_wake_rake_locations)/1000
total_wake_rake_locations=np.array(total_wake_rake_locations)/1000
airfoil_taps_location_top=np.array([[i[1],i[2]] for i in airfoil_pressure_tap_coordinates[:25]])/100
airfoil_taps_location_bottom=np.array([[i[1],i[2]] for i in airfoil_pressure_tap_coordinates[25:]])/100

"""print("airfoil pressure taps:", airfoil_pressure_tap_coordinates)
print("wake rake total ports:", total_wake_rake_locations)
print("wake rake static ports:", static_wake_rake_locations)
print("pitot-static ports:", pitot_static_ports)"""

q_inf = rc.calculate_reference_dynamic_pressure(sum(delta_pb_value) / n_measurements)
p_inf = rc.calculate_reference_static_pressure(sum(total_pressure_pivot) / n_measurements, q_inf)

print(pc.pressure_coefficient(q_inf, p_inf, pressure_taps, 5, airfoil_pressure_tap_coordinates, alpha_array))  #if you change the number you cahnge which experemient you are calucalting the cp for

plot_forces()