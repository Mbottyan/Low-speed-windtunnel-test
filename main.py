import csv
import numpy as np
from numpy import cos, sin
import reference_condtions as rc



    
    


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

