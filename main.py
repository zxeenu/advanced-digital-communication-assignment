import numpy as np
import matplotlib.pyplot as plt
import os
import time
from rich import print

# GLOBALS
DOWNLOAD_PATH = "exports"

# UTILS
def get_timestamp():
    timestamp = str(int(time.time()))
    return timestamp

def task_1_bpsk_simulation():
    """Task 1: BPSK Simulation over AWGN Channel"""

    # 0. Params
    num_bits=10_000

    print(f"\n[green]task_1_bpsk_simulation | num_bits: {num_bits}[/green]")
    rng = np.random.default_rng()

    bits = rng.integers(0, 2, size=num_bits)

    symbols = 2 * bits - 1
    print(f"[BPSK]   Symbols: {symbols}\n")

    snr_db_values = np.array([0, 2, 4, 6, 8, 10, 12])
    ber_values = []

    for snr_db in snr_db_values:

        snr_linear = 10 ** (snr_db / 10)

        noise_std = np.sqrt(1 / (2 * snr_linear))

        noise = noise_std * rng.standard_normal(num_bits)
        received = symbols + noise

        print(f"[BPSK]   Received: {received}")
        detected_bits = (received > 0).astype(np.uint8) 
        print(f"[BPSK]   Detected Bits: {detected_bits}")

        ber = np.mean(detected_bits != bits)
        ber_values.append(ber)

        print(f"[BPSK]   SNR = {snr_db} dB -> BER = {ber:.6f}\n")

    plt.figure()
    plt.semilogy(snr_db_values, ber_values, marker='o')
    plt.title("BPSK over AWGN - BER vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both")
    plt.tight_layout()

    export_path = f"{DOWNLOAD_PATH}/task_1_ber_vs_snr_{get_timestamp()}.png"
    plt.savefig(export_path, dpi=300)
    print(f"Plot saved as {export_path}")
    print(f"[green]task_1 end[/green]")
    return

def task_2_comparison():
    """Task 2: Compare BPSK, QPSK and 16-QAM over AWGN"""

    num_bits = 10_000

    print(f"\n[green]task_2_comparison | num_bits: {num_bits}[/green]")
    rng = np.random.default_rng()
 
    snr_db_values = np.array([0, 2, 4, 6, 8, 10, 12])
 
    bits_bpsk = rng.integers(0, 2, size=num_bits)
 
    symbols_bpsk = 2 * bits_bpsk - 1
 
    ber_bpsk = []
    for snr_db in snr_db_values:
        snr_linear = 10 ** (snr_db / 10)
        noise_std = np.sqrt(1 / (2 * snr_linear))
 
        noise = noise_std * rng.standard_normal(num_bits)
        received = symbols_bpsk + noise
 
        detected = (received > 0).astype(np.uint8)
 
        ber = np.mean(detected != bits_bpsk)
        ber_bpsk.append(ber)
        print(f"[BPSK]   SNR = {snr_db} dB -> BER = {ber:.6f}")
 
    bits_qpsk = rng.integers(0, 2, size=num_bits)
 
    b = bits_qpsk.reshape(-1, 2)
    I = 1 - 2 * b[:, 0].astype(float)
    Q = 1 - 2 * b[:, 1].astype(float)
    symbols_qpsk = (I + 1j * Q) / np.sqrt(2)
    num_symbols_qpsk = len(symbols_qpsk)
 
    ber_qpsk = []
    for snr_db in snr_db_values:
        snr_linear = 10 ** (snr_db / 10)
        es_n0 = 2 * snr_linear
        noise_std = np.sqrt(1 / (2 * es_n0))
 
        noise = noise_std * (rng.standard_normal(num_symbols_qpsk)
                             + 1j * rng.standard_normal(num_symbols_qpsk))
        received = symbols_qpsk + noise
 
        I_bits = (np.real(received) < 0).astype(np.uint8)
        Q_bits = (np.imag(received) < 0).astype(np.uint8)
        detected = np.column_stack([I_bits, Q_bits]).ravel()
 
        ber = np.mean(detected != bits_qpsk)
        ber_qpsk.append(ber)
        print(f"[QPSK]   SNR = {snr_db} dB -> BER = {ber:.6f}")
 
    bits_qam = rng.integers(0, 2, size=num_bits)
 
    gray_map = {(0,0): 3.0, (0,1): 1.0, (1,1): -1.0, (1,0): -3.0}
 
    b4 = bits_qam.reshape(-1, 4)
    I_qam = np.array([gray_map[(row[0], row[1])] for row in b4])
    Q_qam = np.array([gray_map[(row[2], row[3])] for row in b4])
    symbols_qam = (I_qam + 1j * Q_qam) / np.sqrt(10)
    num_symbols_qam = len(symbols_qam)
 
    level_to_bits = {v: k for k, v in gray_map.items()}
    pam_levels = np.array([-3.0, -1.0, 1.0, 3.0])
    boundaries = np.array([-2.0, 0.0, 2.0])
 
    def slice_axis(values):
        """Slice a real-valued axis to nearest 4-PAM level, return bit pairs."""
        indices = np.searchsorted(boundaries, values)
        quantized = pam_levels[indices]
        bits_out = []
        for q in quantized:
            bits_out.extend(level_to_bits[q])
        return np.array(bits_out, dtype=np.uint8)
 
    ber_qam = []
    for snr_db in snr_db_values:
        snr_linear = 10 ** (snr_db / 10)
        es_n0 = 4 * snr_linear
        noise_std = np.sqrt(1 / (2 * es_n0))
 
        noise = noise_std * (rng.standard_normal(num_symbols_qam)
                             + 1j * rng.standard_normal(num_symbols_qam))
        received = symbols_qam + noise
 
        r_scaled = received * np.sqrt(10)
        I_bits_qam = slice_axis(np.real(r_scaled))
        Q_bits_qam = slice_axis(np.imag(r_scaled))
        detected = np.column_stack(
            [I_bits_qam.reshape(-1, 2), Q_bits_qam.reshape(-1, 2)]
        ).ravel()
 
        ber = np.mean(detected != bits_qam)
        ber_qam.append(ber)
        print(f"[16-QAM] SNR = {snr_db} dB -> BER = {ber:.6f}")
 
    plt.figure()
    plt.semilogy(snr_db_values, ber_bpsk,  marker='o', label='BPSK')
    plt.semilogy(snr_db_values, ber_qpsk,  marker='s', label='QPSK')
    plt.semilogy(snr_db_values, ber_qam,   marker='^', label='16-QAM')
    plt.title("BPSK vs QPSK vs 16-QAM over AWGN - BER vs SNR")
    plt.xlabel("SNR - Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    export_path = f"{DOWNLOAD_PATH}/task_2_comparison_ber_vs_snr_{get_timestamp()}.png"
    plt.savefig(export_path, dpi=300)
    print(f"Plot saved as {export_path}")

    print(f"[green]task_2 end[/green]")
    return

def task_3_hamming():
    """Task 3: Hamming (7,4) Error Control Coding"""

    error_position: None | int = 6 

    print(f"\n[green]task_3_hamming_code[/green]")
 
    m = np.array([1, 0, 1, 1])
    print(f"Original message       : {m}")
    print(
        "Labeled message        :",
        " ".join(f"m{i}={bit}" for i, bit in enumerate(m, start=1))
    )
 
    m1, m2, m3, m4 = m
 
    p1 = m1 ^ m2 ^ m4
    p2 = m1 ^ m3 ^ m4
    p4 = m2 ^ m3 ^ m4
 
    codeword = np.array([p1, p2, m1, p4, m2, m3, m4])
    print(f"\nEncoded codeword (7-bit): {codeword}")
    labels = ["p1", "p2", "m1", "p4", "m2", "m3", "m4"]
    print(
        "Labeled codeword       :",
        " ".join(f"{label}={bit}" for label, bit in zip(labels, codeword))
    )

    print(f"\nParity bits            : p1={p1}  p2={p2}  p4={p4}")
 
    received = codeword.copy()

    if error_position is not None:
        received[error_position] ^= 1   
        print(
            f"\n[red]Error injected at bit position {error_position} "
            f"(0-indexed, {labels[error_position]})[/red]"
        )
        print(f"[red]Received codeword      : {received}[/red]")
        print(
            "[red]Labeled received       :",
            " ".join(f"{label}={bit}" for label, bit in zip(labels, received))
        )
        print(
            f"\n[red]Injected error "
            f"(position {error_position + 1} (1-indexed) in Hamming codeword, "
            f"index {error_position} in Python (0-indexed))[/red]"
        )
    else:
        print("No error injected.")

    r1, r2, r3, r4, r5, r6, r7 = received

    s1 = r1 ^ r3 ^ r5 ^ r7     # checks bit positions 0,2,4,6
    s2 = r2 ^ r3 ^ r6 ^ r7     # checks bit positions 1,2,5,6
    s4 = r4 ^ r5 ^ r6 ^ r7     # checks bit positions 3,4,5,6
 
    syndrome = s4 * 4 + s2 * 2 + s1 * 1
    print(f"\nSyndrome bits          : s1={s1}  s2={s2}  s4={s4}")
    print(f"Syndrome value         : {syndrome}  (binary: {s4}{s2}{s1})")
 
    corrected = received.copy()
    if syndrome == 0:
        print("No error detected.")
    else:
        error_index = syndrome - 1
        print(
            f"[red]Error detected "
            f"(Hamming position {syndrome}, "
            f"Python index {error_index}, {labels[error_index]})[/red]"
        )
        corrected[error_index] ^= 1
    
    print(f"Corrected codeword     : {corrected}")
 
    recovered = corrected[[2, 4, 5, 6]]
    print(f"\nOriginal message       : {m}")
    print(f"Recovered message      : {recovered}")
 
    match = np.array_equal(m, recovered)
    print(f"Match                  : {match}")
    assert match, "ERROR: recovered message does not match original!"
    print("\nTask 3 complete: Hamming (7,4) encode -> error -> correct -> recover 😀")

    print(f"[green]task_3 end[/green]")
    return

def task_4_ofdm_simulation():
    """Task 4: Simple OFDM Transmitter Simulation"""

    print(f"\n[green]task_4_ofdm_simulation[/green]")

    rng = np.random.default_rng()

    num_bits = 8000
    num_subcarriers = 8
    cp_length = 2 

    print(f"Subcarriers: {num_subcarriers}, CP length: {cp_length}")

    bits = rng.integers(0, 2, size=num_bits)

    bits = bits[: (len(bits) // 2) * 2]

    bit_pairs = bits.reshape(-1, 2)

    I = 1 - 2 * bit_pairs[:, 0]
    Q = 1 - 2 * bit_pairs[:, 1]

    qpsk_symbols = (I + 1j * Q) / np.sqrt(2)

    print(f"QPSK symbols generated: {len(qpsk_symbols)}")

    num_ofdm_symbols = len(qpsk_symbols) // num_subcarriers
    qpsk_symbols = qpsk_symbols[:num_ofdm_symbols * num_subcarriers]

    ofdm_matrix = qpsk_symbols.reshape(num_ofdm_symbols, num_subcarriers)

    time_domain = np.fft.ifft(ofdm_matrix, axis=1, norm='ortho')

    cp = time_domain[:, -cp_length:]
    ofdm_with_cp = np.concatenate([cp, time_domain], axis=1)

    tx_signal = ofdm_with_cp.flatten()

    plt.figure()
    plt.scatter(np.real(qpsk_symbols[:500]), np.imag(qpsk_symbols[:500]))
    plt.title("QPSK Constellation")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.tight_layout()

    path1 = f"{DOWNLOAD_PATH}/task_4_qpsk_constellation_{get_timestamp()}.png"
    plt.savefig(path1, dpi=300)
    print(f"Saved: {path1}")

    plt.figure()
    plt.plot(np.real(time_domain.flatten()[:500]))
    plt.title("OFDM Time-Domain Signal (Real Part)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()

    path2 = f"{DOWNLOAD_PATH}/task_4_ofdm_time_{get_timestamp()}.png"
    plt.savefig(path2, dpi=300)
    print(f"Saved: {path2}")

    plt.figure()
    plt.plot(np.real(tx_signal[:500]))
    plt.title("OFDM Signal with Cyclic Prefix")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()

    path3 = f"{DOWNLOAD_PATH}/task_4_ofdm_cp_{get_timestamp()}.png"
    plt.savefig(path3, dpi=300)
    print(f"Saved: {path3}")

    _, axes = plt.subplots(3, 1, figsize=(8, 12)) 

    axes[0].scatter(np.real(qpsk_symbols[:500]), np.imag(qpsk_symbols[:500]), s=5)
    axes[0].set_title("QPSK Constellation")
    axes[0].set_xlabel("In-phase")
    axes[0].set_ylabel("Quadrature")
    axes[0].grid(True)

    axes[1].plot(np.real(time_domain.flatten()[:500]))
    axes[1].set_title("OFDM Time-Domain Signal (Real Part)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(np.real(tx_signal[:500]))
    axes[2].set_title("OFDM Signal with Cyclic Prefix")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    plt.tight_layout()
    combined_path = f"{DOWNLOAD_PATH}/task_4_combined_{get_timestamp()}.png"
    plt.savefig(combined_path, dpi=300)
    print(f"Saved: {combined_path}")

    print(f"[green]task_4 end[/green]")
    return

def task_5_ofdm_receiver_ber():
    """Bonus Task: OFDM Receiver + BER over AWGN"""

    print(f"\n[green]task_5_ofdm_receiver_ber[/green]")

    rng = np.random.default_rng()

    num_bits = 8000
    num_subcarriers = 8
    cp_length = 2
    snr_db_values = np.array([0, 2, 4, 6, 8, 10, 12])

    bits = rng.integers(0, 2, size=(num_bits // 2) * 2)
    bit_pairs = bits.reshape(-1, 2)

    I = 1 - 2 * bit_pairs[:, 0]
    Q = 1 - 2 * bit_pairs[:, 1]
    qpsk_symbols = (I + 1j * Q) / np.sqrt(2)

    num_ofdm_symbols = len(qpsk_symbols) // num_subcarriers
    qpsk_symbols = qpsk_symbols[:num_ofdm_symbols * num_subcarriers]
    ofdm_matrix = qpsk_symbols.reshape(num_ofdm_symbols, num_subcarriers)

    time_domain = np.fft.ifft(ofdm_matrix, axis=1, norm='ortho')

    cp = time_domain[:, -cp_length:]
    ofdm_with_cp = np.concatenate([cp, time_domain], axis=1)

    tx_signal = ofdm_with_cp.reshape(-1)

    ber_values = []

    for snr_db in snr_db_values:

        snr_linear = 10 ** (snr_db / 10)

        noise_std = np.sqrt(1 / (2 * snr_linear))

        noise = noise_std * (
            rng.standard_normal(len(tx_signal))
            + 1j * rng.standard_normal(len(tx_signal))
        )

        rx_signal = tx_signal + noise

        rx_blocks = rx_signal.reshape(num_ofdm_symbols, num_subcarriers + cp_length)

        rx_no_cp = rx_blocks[:, cp_length:]

        freq_domain = np.fft.fft(rx_no_cp, axis=1, norm='ortho')

        received_symbols = freq_domain.reshape(-1)

        I_hat = (np.real(received_symbols) < 0).astype(np.uint8)
        Q_hat = (np.imag(received_symbols) < 0).astype(np.uint8)

        detected_bits = np.column_stack([I_hat, Q_hat]).ravel()

        ber = np.mean(detected_bits != bits[:len(detected_bits)])
        ber_values.append(ber)

        print(f"SNR = {snr_db} dB -> BER = {ber:.6f}")

    plt.figure()
    plt.semilogy(snr_db_values, ber_values, marker="o")
    plt.title("OFDM BER over AWGN")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True, which="both")
    plt.tight_layout()

    path = f"{DOWNLOAD_PATH}/task_5_ofdm_ber_{get_timestamp()}.png"
    plt.savefig(path, dpi=300)

    print(f"Saved BER plot: {path}")
    print(f"[green]task_5 end[/green]")
    return

def main():
    # SETUP
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    # SIMULATIONS
    print(f"[green]Starting simulations...[/green]")
    task_1_bpsk_simulation()
    task_2_comparison()
    task_3_hamming()
    task_4_ofdm_simulation()
    task_5_ofdm_receiver_ber()
    return


if __name__ == "__main__":
    main()