# ==========================================
# DATI HARDCODED (Inserimento con unità personalizzate)
# ==========================================

# -- Variabili Radio / Rete --
P_tx = 0.2           # P_{tx}: Potenza radio in Watt (W)
R_link_mbps = 1000.0 # R_{link}: Bit rate del link in Mbit/s
E_fixed_mj = 0.2     # E_{fixed}: Overhead fisso in millijoule (mJ)
S_probe = 200.0      # S_{probe}: Dimensione del probe in Byte

# -- Variabili di Job/CPU --
P_cpu = 0.8          # P_{cpu}: Potenza CPU in Watt (W)
t_exec = 0.0075        # t_{exec}: Durata media di esecuzione in secondi (s)
S_job_kb = 2.0      # S_{job}: Dimensione media dei payload job in Kilobyte (KB)

# -- Variabili di Probing --
N_probe = 3.0        # N_{probe}: Numero di probe inviati per job

# ==========================================
# CONVERSIONI AUTOMATICHE
# ==========================================

R_link = R_link_mbps * 1_000_000  # Converte Mbit/s in bit/s
E_fixed = E_fixed_mj / 1000.0     # Converte mJ in Joule (J)
S_job = S_job_kb * 1024.0         # Converte KB in Byte (assumendo 1 KB = 1024 Byte)

# ==========================================
# CALCOLI
# ==========================================

# 1. Calcolo di E_probe (Joule)
# Nota: S_probe * 8 per convertire i Byte in bit
E_probe = (P_tx * (S_probe * 8) / R_link) + E_fixed

# 2. Calcolo di E_job (Joule)
# Nota: S_job * 8 per convertire i Byte in bit
E_job = (P_cpu * t_exec) + (P_tx * (S_job * 8) / R_link)

# 3. Calcolo di rho (Quota adimensionale)
rho = (N_probe * E_probe) / E_job

# 4. Calcolo di rho in percentuale (%)
rho_perc = rho * 100

# ==========================================
# OUTPUT RISULTATI
# ==========================================

print("-" * 34)
print(" RISULTATI DEI CALCOLI ENERGETICI")
print("-" * 34)
print(f"E_probe : {E_probe:.8f} J")
print(f"E_job   : {E_job:.8f} J")
print(f"rho     : {rho:.8f}")
print(f"rho %   : {rho_perc:.4f} %")
print("-" * 34)
