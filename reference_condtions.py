def calculate_air_density(p_atm, t_celsius, m_air=0.0289652, R=8.314462618):
    # air density in kg/m^3
    T = t_celsius + 273.15
    return m_air * p_atm / (R * T)

def calculate_dynamic_viscosity(t_celsius, T0=273.15, mu0=1.716e-5, S=110.4):
    # dynamic viscosity in kg/(m*s)
    T = t_celsius + 273.15
    return mu0 * ((T / T0) ** 1.5) * ((T0 + S) / (T + S))

def calculate_reference_dynamic_pressure(dp):
    # dynamic pressure in Pa
    return 0.211804 + 1.928442 * dp + 1.879374e-4 * (dp ** 2)

def calculate_reference_static_pressure(pt, q):
    # static pressure in Pa
    return pt - q
