def pressure_coefficient(q_inf, p_inf, pressure_taps, nth_measurement, coordinates):
    out = []
    for i in range(len(coordinates)):
        out.append((coordinates[i][1], coordinates[i][2], (pressure_taps[i][nth_measurement] - p_inf) / q_inf))
    plot_cp(out)
    return out


def plot_cp(data):
    import matplotlib.pyplot as plt
    upper = sorted([d for d in data if d[1] >= 0], key=lambda v: v[0])
    lower = sorted([d for d in data if d[1] < 0], key=lambda v: v[0])

    plt.figure(figsize=(10, 6))

    # Plot upper surface
    if upper:
        plt.plot([d[0] for d in upper], [d[2] for d in upper], label='Upper Surface', linewidth=2, marker='o', color='tab:orange')

    # Plot lower surface
    if lower:
        plt.plot([d[0] for d in lower], [d[2] for d in lower], label='Lower Surface', linewidth=2, marker='o', color='tab:blue')

    plt.axhline(0, color='gray', linewidth=2, linestyle='--')
    
    plt.xlabel('X/c', fontsize=14, fontweight='bold')
    plt.ylabel('Cp', fontsize=14, fontweight='bold')
    plt.title('Cp vs X/c', fontsize=16, fontweight='bold')
    
    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.grid(True, alpha=0.5)
    
    # Invert Y axis so negative Cp is up
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

