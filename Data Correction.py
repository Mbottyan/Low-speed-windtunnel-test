import pandas as pd
import math


# input
RAW_DATA_FILE = 'data files/Forces (Cl&Cd vs AoA)/forces.csv'
# output
OUTPUT_FILE = 'data files/Forces (Cl&Cd vs AoA)/corrected_forces.csv'


# constants (in meters)
h = 0.6   #from floor to ceiling (h)
c  = 0.16  #airfoil chord (c)
t_max = c * 0.104 #maximum thickness(t)

#   variables
corrected_alpha = []
corrected_cl = []
corrected_cd = []
corrected_cm = []

# ______________read the data___________________
try:
    df = pd.read_csv(RAW_DATA_FILE)
except FileNotFoundError:
    print(f"file was not found.")
    exit()
alpha_geometric = df['alpha'].tolist()
raw_cl = df['cl'].tolist()
raw_cd = df['cd'].tolist()
raw_cm = df['cm'].tolist()

# _____________calculations______________________

# solid blockage factor
epsilon_sb = (math.pi**2 / 12) * (t_max * c) / (h ** 2)

# lift interference factor
sigma = (math.pi**2 / 48) * ((c / h) ** 2)

print(f"{'Raw Alpha':<12} | {'Corr Alpha':<12} | {'Raw Cl':<10} | {'Corr Cl':<10} | {'Raw Cd':<10} | {'Corr Cd':<10} | {'Raw Cm':<10} | {'Corr Cm':<10}")
print("-" * 100)


for i in range(len(alpha_geometric)):

    # go over every datapoint
    alpha_raw = alpha_geometric[i]
    cl_raw = raw_cl[i]
    cd_raw = raw_cd[i]
    cm_raw = raw_cm[i]

    # wake blockage factor
    epsilon_wb = 0.25 * (c / h) * cd_raw

    # total blockage factor
    epsilon_total = epsilon_sb + epsilon_wb

    # ________corretions_________
    correction_factor = (1 - 2 * epsilon_total)

    cl_corrected = cl_raw * correction_factor
    cd_corrected = cd_raw * correction_factor
    cm_corrected = (cm_raw * correction_factor) + (0.25 * sigma * cl_corrected)

    alpha_corrected = alpha_raw + math.degrees((sigma / (2 * math.pi)) * (cl_raw + 4 * cm_raw))

    # output
    corrected_alpha.append(round(alpha_corrected, 3))
    corrected_cl.append(round(cl_corrected, 4))
    corrected_cd.append(round(cd_corrected, 5))
    corrected_cm.append(round(cm_corrected, 5))
    print(f"{alpha_raw:<12.3f} | {alpha_corrected:<12.3f} | {cl_raw:<10.4f} | {cl_corrected:<10.4f} | {cd_raw:<10.5f} | {cd_corrected:<10.5f} | {cm_raw:<10.5f} | {cm_corrected:<10.5f}")


# _____________save corrected data______________________
corrected_data = pd.DataFrame({
    'alpha': corrected_alpha,
    'cl': corrected_cl,
    'cd': corrected_cd,
    'cm': corrected_cm
})
corrected_data.to_csv(OUTPUT_FILE, index=False)
