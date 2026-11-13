# Quantum_girsanov_radon



import random
import numpy as np

random.seed(39)
np.random.seed(39)

from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_BALLS = 39
DRAW_SIZE = 7
N_QUBITS = 6
N_STATES = 2 ** N_QUBITS


def load_draws(csv_path: Path) -> np.ndarray:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            nums = [int(x) for x in parts[:DRAW_SIZE]]
            if len(nums) != DRAW_SIZE:
                raise ValueError(f"Neispravan red u {csv_path}: {line}")
            if any(n < 1 or n > NUM_BALLS for n in nums) or len(set(nums)) != DRAW_SIZE:
                raise ValueError(f"Nelegalna kombinacija u {csv_path}: {nums}")
            rows.append(sorted(nums))
    if len(rows) < 100:
        raise ValueError(f"Premalo kola u {csv_path}: {len(rows)}")
    return np.array(rows, dtype=np.int32)


def draws_to_binary_matrix(draws: np.ndarray) -> np.ndarray:
    t_len = len(draws)
    mat = np.zeros((t_len, NUM_BALLS), dtype=np.float64)
    rows_idx = np.arange(t_len)[:, None]
    mat[rows_idx, draws - 1] = 1.0
    return mat


def compute_girsanov_increments_and_base_measure(
    binary_mat: np.ndarray, short_win: int = 8, long_win: int = 34
) -> tuple[np.ndarray, np.ndarray]:
    t_len, n_balls = binary_mat.shape
    base_p = np.zeros((t_len, n_balls), dtype=np.float64)
    girsanov_signal = np.zeros((t_len, n_balls), dtype=np.float64)

    cum_counts = np.zeros(n_balls, dtype=np.float64)
    eps = 1e-6

    for t in range(t_len):
        cum_counts += binary_mat[t]
        p_t = (cum_counts + 1.0) / float(t + 1 + n_balls)
        p_t /= np.sum(p_t)
        base_p[t] = p_t

        start_s = max(0, t + 1 - short_win)
        start_l = max(0, t + 1 - long_win)

        sub_s = binary_mat[start_s : t + 1]
        sub_l = binary_mat[start_l : t + 1]

        mean_s = np.mean(sub_s, axis=0)
        mean_l = np.mean(sub_l, axis=0)
        vol_l = np.std(sub_l, axis=0) + eps

        drift_diff = mean_s - mean_l
        girsanov_signal[t] = drift_diff / vol_l

    return base_p, girsanov_signal


def build_radon_nikodym_amplitudes(
    base_p_t: np.ndarray, girsanov_t: np.ndarray, theta_39: np.ndarray
) -> np.ndarray:
    log_z = theta_39 * girsanov_t - 0.5 * (theta_39 ** 2)
    log_z = np.clip(log_z, -6.0, 6.0)
    z_39 = np.exp(log_z)

    q_39 = base_p_t * z_39
    q_39 = np.maximum(q_39, 1e-12)
    q_39 /= np.sum(q_39)

    amp_64 = np.zeros(N_STATES, dtype=np.float64)
    amp_64[:NUM_BALLS] = np.sqrt(q_39)

    rem_mass = max(0.0, 1.0 - float(np.sum(amp_64[:NUM_BALLS] ** 2)))
    tail_slots = N_STATES - NUM_BALLS
    if rem_mass > 0.0 and tail_slots > 0:
        amp_64[NUM_BALLS:] = np.sqrt(rem_mass / float(tail_slots))

    norm = float(np.linalg.norm(amp_64))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Neispravna norma kvantnih amplituda.")
    amp_64 /= norm
    return amp_64


def quantum_measure_transform(
    amp_64: np.ndarray, q_params: np.ndarray
) -> np.ndarray:
    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(amp_64.tolist(), list(range(N_QUBITS)))

    idx = 0
    for _ in range(2):
        for q in range(N_QUBITS):
            qc.ry(float(q_params[idx]), q)
            idx += 1
            qc.rz(float(q_params[idx]), q)
            idx += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
        qc.cx(N_QUBITS - 1, 0)

    sv = Statevector.from_instruction(qc)
    probs_64 = np.abs(sv.data) ** 2

    dist_39 = probs_64[:NUM_BALLS].copy()
    dist_39 = np.maximum(dist_39, 1e-12)
    dist_39 /= np.sum(dist_39)
    return dist_39


def predict_step_distribution(
    base_p_t: np.ndarray, girsanov_t: np.ndarray, full_params: np.ndarray
) -> np.ndarray:
    theta_39 = full_params[:NUM_BALLS]
    q_params = full_params[NUM_BALLS:]
    amp_64 = build_radon_nikodym_amplitudes(base_p_t, girsanov_t, theta_39)
    return quantum_measure_transform(amp_64, q_params)


def train_girsanov_quantum_model(
    base_p_seq: np.ndarray,
    girsanov_seq: np.ndarray,
    target_binary_seq: np.ndarray,
    reg_lambda: float = 2e-3,
    max_iter: int = 45,
) -> np.ndarray:
    n_samples = len(base_p_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)

    step = max(1, n_samples // 180)
    sub_p = base_p_seq[::step]
    sub_g = girsanov_seq[::step]
    sub_tgt = target_dist[::step]
    sub_len = len(sub_p)

    n_q_params = 2 * N_QUBITS * 2
    n_total_params = NUM_BALLS + n_q_params

    rng = np.random.RandomState(39)
    init_params = np.zeros(n_total_params, dtype=np.float64)
    init_params[:NUM_BALLS] = rng.uniform(-0.15, 0.15, size=NUM_BALLS)
    init_params[NUM_BALLS:] = rng.uniform(-0.20, 0.20, size=n_q_params)

    def loss_fn(params: np.ndarray) -> float:
        total_loss = 0.0
        for i in range(sub_len):
            pred_dist = predict_step_distribution(sub_p[i], sub_g[i], params)
            total_loss -= float(np.sum(sub_tgt[i] * np.log(pred_dist)))
        total_loss /= float(sub_len)
        total_loss += reg_lambda * float(np.sum(params ** 2))
        return float(total_loss)

    res = minimize(
        loss_fn,
        init_params,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-5},
    )
    return res.x


def evaluate_distribution(
    base_p_seq: np.ndarray,
    girsanov_seq: np.ndarray,
    target_binary_seq: np.ndarray,
    params: np.ndarray,
) -> tuple[float, float]:
    n_samples = len(base_p_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)
    nll = 0.0
    hits = 0.0
    for i in range(n_samples):
        pred_dist = predict_step_distribution(base_p_seq[i], girsanov_seq[i], params)
        nll -= float(np.sum(target_dist[i] * np.log(pred_dist)))
        top7 = np.argsort(pred_dist)[-DRAW_SIZE:]
        actual = np.flatnonzero(target_binary_seq[i] > 0.5)
        hits += float(len(set(top7.tolist()).intersection(actual.tolist())))
    return nll / float(n_samples), hits / float(n_samples)


def select_next_combination(dist_39: np.ndarray) -> list[int]:
    if dist_39.shape != (NUM_BALLS,) or not np.all(np.isfinite(dist_39)):
        raise ValueError("Neispravna izlazna raspodela modela.")
    if abs(float(np.sum(dist_39)) - 1.0) > 1e-5:
        raise ValueError("Raspodela nije normalizovana.")
    top7_idx = np.argsort(dist_39)[-DRAW_SIZE:]
    combo = sorted(int(i + 1) for i in top7_idx)
    if len(combo) != DRAW_SIZE or len(set(combo)) != DRAW_SIZE:
        raise ValueError("Nelegalna izlazna kombinacija.")
    return combo


def run_pipeline_for_csv(csv_path: Path) -> None:
    draws = load_draws(csv_path)
    binary_mat = draws_to_binary_matrix(draws)
    t_len = len(binary_mat)

    n_trans = t_len - 1
    train_end = int(n_trans * 0.70)
    val_end = int(n_trans * 0.85)

    candidate_windows = (
        (6, 26),
        (10, 42),
    )

    best_win = next(iter(candidate_windows))
    best_val_nll = float("inf")
    best_val_hits = 0.0
    best_test_nll = float("inf")
    best_test_hits = 0.0

    for short_w, long_w in candidate_windows:
        base_p, girs_sig = compute_girsanov_increments_and_base_measure(
            binary_mat, short_win=short_w, long_win=long_w
        )

        p_train = base_p[:train_end]
        g_train = girs_sig[:train_end]
        y_train = binary_mat[1 : train_end + 1]

        p_val = base_p[train_end:val_end]
        g_val = girs_sig[train_end:val_end]
        y_val = binary_mat[train_end + 1 : val_end + 1]

        params_cv = train_girsanov_quantum_model(p_train, g_train, y_train, max_iter=35)
        val_nll, val_hits = evaluate_distribution(p_val, g_val, y_val, params_cv)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_hits = val_hits
            best_win = (short_w, long_w)

            p_test = base_p[val_end:n_trans]
            g_test = girs_sig[val_end:n_trans]
            y_test = binary_mat[val_end + 1 : n_trans + 1]
            best_test_nll, best_test_hits = evaluate_distribution(
                p_test, g_test, y_test, params_cv
            )

    best_short, best_long = best_win
    base_p_final, girs_final = compute_girsanov_increments_and_base_measure(
        binary_mat, short_win=best_short, long_win=best_long
    )

    p_full = base_p_final[:n_trans]
    g_full = girs_final[:n_trans]
    y_full = binary_mat[1 : n_trans + 1]
    final_params = train_girsanov_quantum_model(p_full, g_full, y_full, max_iter=50)

    next_dist = predict_step_distribution(base_p_final[-1], girs_final[-1], final_params)
    next_combo = select_next_combination(next_dist)

    print(f"CSV: {csv_path.name} (kola: {t_len})")
    print(f"Validacija -> NLL: {best_val_nll:.4f} | Prosek pogodaka: {best_val_hits:.3f}")
    print(f"Test       -> NLL: {best_test_nll:.4f} | Prosek pogodaka: {best_test_hits:.3f}")
    print(f"NEXT       -> {next_combo}")
    print("-" * 52)


def main() -> None:
    print("Metoda: Girsanov Drift Shift + Radon-Nikodym Quantum Measure Transform (6 qubita)")
    print("Posebnost: Eksponencijalni martingale izvod mere dQ/dP u kvantnim amplitudama")
    print("=" * 52)
    for csv_path in FILES:
        run_pipeline_for_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Girsanov Drift Shift + Radon-Nikodym Quantum Measure Transform (6 qubita)
Posebnost: Eksponencijalni martingale izvod mere dQ/dP u kvantnim amplitudama
====================================================

CSV: loto7_4718_k90.csv (kola: 4718)
Validacija -> NLL: 3.6881 | Prosek pogodaka: 1.257
Test       -> NLL: 3.6922 | Prosek pogodaka: 1.220
NEXT       -> [2, x, 31, y, 33, z, 39]
----------------------------------------------------

CSV: loto7_4718_k90_loto_2981.csv (kola: 2981)
Validacija -> NLL: 3.6920 | Prosek pogodaka: 1.262
Test       -> NLL: 3.6936 | Prosek pogodaka: 1.233
NEXT       -> [1, x, 11, y, 26, z, 32]
----------------------------------------------------

CSV: loto7_4718_k90_loto_plus_1737.csv (kola: 1737)
Validacija -> NLL: 3.6830 | Prosek pogodaka: 1.208
Test       -> NLL: 3.6823 | Prosek pogodaka: 1.322
NEXT       -> [1, x, 4, y, 28, z, 33]
----------------------------------------------------
"""



"""
The Radon-Nikodym Theorem(Fuel): which gives us a formal way to reweight probabilities. Think of it as a density function that tilts the odds from “what actually happens” to “what we pretend happens for pricing.”
The Girsanov Theorem (Engine) : which tells us exactly how to change the drift of a random process (like a stock price) without altering its volatility. It's the engine under the hood of every measure change in continuous-time finance.


1. Radon-Nikodym teorema (Gorivo / Promena mere): Matematički alat koji povezuje dve mere verovatnoće \(P\) (empirijska/stvarna mera iz istorije) i \(Q\) (ciljna/uslovna mera za sledeći korak) preko Radon-Nikodym izvoda \(Z_t = \frac{dQ}{dP}\). U praksi to je pozitivna funkcija gustine (faktor ponovnog ponderisanja) koja preslikava polaznu raspodelu u novu raspodelu bez narušavanja osnovnog prostora događaja.
2. Girsanovljeva teorema (Motor / Pomeranje drifta): Pokazuje kako promena mere putem eksponencijalnog martingala \(Z_t = \exp\left(\int_0^t \theta_s \, dW_s - \frac{1}{2}\int_0^t \theta_s^2 \, ds\right)\) menja drift (usmereni trend/pomeraj) stohastičkog procesa, dok njegova lokalna volatilnost (struktura disperzije/varijanse) ostaje očuvana.

Čist matematički okvir za regresiju raspodele bez svođenja na proste istorijske frekvencije i bez nametanja normalne ili uniformne raspodele.

• Dinamičko pomeranje drifta (Girsanov) umesto statičke frekvencije: Umesto da pretpostavimo da svi brojevi imaju fiksnu verovatnoću, istorija izvlačenja do kola \(t\) definiše empirijsku meru \(P_{t}\). Lokalno stanje procesa (višerezolucijski inkrementi i struktura prelaza do \(t\)) određuje vektor tržišne cene rizika / drifta \(\theta _{t}\), koji pomera očekivani smer svakog od 39 kanala uz očuvanje njihove istorijske volatilnosti.
• Radon-Nikodym izvod kao kvantna amplituda (Qiskit, 6 qubita): Eksponencijalni martingale faktor \(Z_t = \frac{dQ_t}{dP_t}\) prirodno je pozitivan i normalizovan, što ga čini direktnim kandidatom za kodiranje u amplitude kvantnog stanja — 6 qubita, 64 bazna stanja. Kvantno kolo (Qiskit Statevector simulator) može da uči parametrizovanu transformaciju mere (kvantni analogon promene mere \(P \to Q\)), dok hronološka validacija i test proveravaju da li pomeranje mere smanjuje negativnu log-verodostojnost (NLL) u odnosu na baznu meru \(P_{t}\).
"""
