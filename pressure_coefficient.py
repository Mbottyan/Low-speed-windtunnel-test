def pressure_coefficient(q_inf, p_inf, pressure_taps, nth_measurement, coordinates):
    out = []
    for i in range(len(coordinates)):
        out.append((coordinates[i][1], coordinates[i][2], (pressure_taps[i][nth_measurement]) / q_inf))
    plot_cp(out)
    return out


def plot_cp(data):
    import matplotlib.pyplot as plt

    # Separate upper (y > 0) and lower (y < 0) surfaces and sort by x
    upper = sorted([d for d in data if d[1] > 0], key=lambda v: v[0])
    lower = sorted([d for d in data if d[1] < 0], key=lambda v: v[0])

    plt.figure(figsize=(10, 6))

    # Plot lower surface (y < 0) - Blue in example
    if lower:
        plt.plot([d[0] for d in lower], [d[2] for d in lower], label='Lower', linewidth=3, color='tab:blue', alpha=0.8)

    # Plot upper surface (y > 0) - Orange in example
    if upper:
        plt.plot([d[0] for d in upper], [-1 * d[2] for d in upper], label='Upper', linewidth=3, color='tab:orange', alpha=0.8)

    plt.axhline(0, color='gray', linewidth=2, linestyle='--')
    
    plt.xlabel('X/c', fontsize=14, fontweight='bold')
    plt.ylabel('Cp', fontsize=14, fontweight='bold')
    plt.title('Cp vs X/c', fontsize=16, fontweight='bold')
    
    plt.legend(fontsize=12, loc='center right', frameon=False)
    plt.grid(True, alpha=0.5)
    
    # Invert Y axis so negative Cp is up (standard aerodynamic plot)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

