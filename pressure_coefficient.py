def pressure_coefficient(P_atm, q_inf, p_inf, pressure_taps, nth_measurement, coordinates, aoa, Plot=True):
    out = []
    for i in range(len(coordinates)):
        out.append((coordinates[i][1], coordinates[i][2], (pressure_taps[nth_measurement][i] + P_atm - p_inf) / q_inf))
    
    # Writing to txt file
    with open("data files/cp_data/experimental/cp_data at {}° aoa.txt".format(aoa[nth_measurement]), "w") as f:
        for x, y, cp in out:
            f.write(f"{x/100:.6f} {cp:.6f}\n")

    if Plot:
        plot_cp(out)
    return out


def plot_cp(data):
    linewidth_major=2
    marker_s_main=5
    axis_font=20
    legend_font=16
    import matplotlib.pyplot as plt
    upper = data[:25]
    lower = data[25:]

    plt.figure(figsize=(10, 6))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha = 0.5)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")

    # Plot upper surface
    if upper:
        plt.plot([d[0] for d in upper], [d[2] for d in upper], label='Upper Surface', linewidth=linewidth_major, marker='o', color='tab:orange',ms=marker_s_main)

    # Plot lower surface
    if lower:
        plt.plot([d[0] for d in lower], [d[2] for d in lower], label='Lower Surface', linewidth=linewidth_major, marker='o', color='tab:blue',ms=marker_s_main)
    
    plt.xlabel('X/c', fontsize=axis_font)
    plt.ylabel(r'$\mathrm{C_p}$', fontsize=axis_font)
    
    plt.legend(fontsize=legend_font, loc='best')
    
    # Invert Y axis so negative Cp is up
    plt.gca().invert_yaxis()
    
    plt.show()

