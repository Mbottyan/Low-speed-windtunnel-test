filename = "raw_Group12_3d.txt"

# lists for data columns
run_number = []
time = []
alpha = []
delta_pb_value = []
p_bar_value = []
temperature = []
force_fx = []
force_fy = []
force_fz = []
rpm = []
density = []

with open(filename, "r", encoding="utf-8") as data_file:
    lines = data_file.readlines()
    # Skip first two lines
    for raw_line in lines[2:]:
        parts = raw_line.split()

        run_number.append(int(parts[0]))
        time.append(parts[1])
        alpha.append(float(parts[2]))
        delta_pb_value.append(float(parts[3]))
        p_bar_value.append(float(parts[4]))
        temperature.append(float(parts[5]))
        force_fx.append(float(parts[6]))
        force_fy.append(float(parts[7]))
        force_fz.append(float(parts[8]))
        rpm.append(float(parts[9]))
        density.append(float(parts[10]))

print("Loaded", len(run_number), "rows from", filename)