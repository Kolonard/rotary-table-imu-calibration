#!/usr/bin/env python3
"""Калибровка одного датчика по записи поворотного стола в формате .csv.

Вход: .csv со столбцами
    timestamp, sys_id, comp_id, time_boot_ms, xacc, yacc, zacc,
    xgyro, ygyro, zgyro, xmag, ymag, zmag, temperature
Выход: оценки смещений и матриц масштаб/неортогональность для гиро и акс.

Раскладка вектора X (12): [d(0:3); Kxx Kyy Kzz (3:6); X7 X8; X9 X10; X11 X12 (6:12)],
то есть  meas - ref = d + dK @ ref,  dK = [[X4, X7, X8], [X9, X5, X10], [X11, X12, X6]]
(нумерация X1..X12 как в MATLAB-версии).

Зависимости: numpy, pandas, matplotlib.
Индексы строк в конфиге и в таблицах — 1-based, включительно (как в MATLAB-версии).
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ===================== НАСТРОЙКИ =====================


@dataclass
class CsvCfg:
    header: bool = True            # есть ли строка заголовка
    sys_id: int | None = None      # None — все строки; число — оставить только этот sys_id
    comp_id: int | None = None     # None — все строки; число — оставить только этот comp_id
    # Масштабы к g и °/с. По умолчанию как в MAVLink SCALED_IMU:
    # акселерометр в мG, гироскоп в мрад/с, температура в сотых долях °C.
    # Если в файле уже g и °/с — поставьте 1.
    acc_scale: float = 1e-3                  # -> g
    gyro_scale: float = math.degrees(1e-3)   # -> °/с
    temp_scale: float = 0.01                 # -> °C


@dataclass
class RangeCfg:
    # unit: "index" — границы в номерах строк (после фильтра sys_id/comp_id и удаления пропусков),
    #       "sec"   — в секундах от первой строки по time_boot_ms
    unit: str = "index"
    start: float = 1
    stop: float = math.inf         # math.inf — до конца записи
    pick: bool = False             # True — выбрать начало и конец мышью (два клика)


@dataclass
class PlateauCfg:
    # ranges   — окна поиска полок, строки [начало, конец]; индексы по ПОЛНОЙ записи (1-based)
    # med      — окно медианного фильтра (подавление выбросов), отсчётов
    # win      — окно локального размаха: полка = место, где в окне win размах <= tol по ВСЕМ осям
    # tol      — допуск локального размаха (ед. измерения)
    # min_len  — минимальная длина полки до обрезки краёв
    # trim_frac, trim_min — обрезка краёв: max(trim_min, trim_frac*длина) с каждой стороны
    # drift_tol — допуск по разности средних первой и второй половины полки
    # vals     — эталонные значения по оси
    # nominal  — K×3 допустимые эталонные векторы; None — привязка по осям к vals
    # snap_tol — макс. отклонение среднего от эталона; дальше — полка отбраковывается
    # norm_tol — для акс: допуск на | |a| - 1 | в g (inf — отключить)
    ranges: list = field(default_factory=lambda: [[1, math.inf]])
    med: int = 9
    win: int = 50
    tol: float = 0.05
    min_len: int = 100
    trim_frac: float = 0.10
    trim_min: int = 10
    drift_tol: float = 0.02
    vals: np.ndarray = field(default_factory=lambda: np.array([-1, -math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 1]))
    nominal: np.ndarray | None = None
    snap_tol: float = 0.10
    norm_tol: float = 0.10


def default_acc_cfg() -> PlateauCfg:
    return PlateauCfg()


def default_grs_cfg() -> PlateauCfg:
    steps = np.array([30 * math.sqrt(2), 60.0])
    return PlateauCfg(
        ranges=[[1, math.inf]], tol=3.0, drift_tol=1.0,
        vals=np.sort(np.unique(np.concatenate([steps, -steps, [0.0]]))),
        snap_tol=3.0, norm_tol=math.inf,
    )


@dataclass
class Config:
    file: str = "131026/IMU_0.csv"
    csv: CsvCfg = field(default_factory=CsvCfg)
    range: RangeCfg = field(default_factory=RangeCfg)
    acc: PlateauCfg = field(default_factory=default_acc_cfg)
    grs: PlateauCfg = field(default_factory=default_grs_cfg)
    plot_seg: bool = True


# ===================== ЗАГРУЗКА ДАННЫХ =====================

# (имя столбца, номер столбца с 1) — запасной вариант, если имени нет в заголовке
_COLS = {
    "sys_id": 2, "comp_id": 3, "time_boot_ms": 4,
    "xacc": 5, "yacc": 6, "zacc": 7, "xgyro": 8, "ygyro": 9, "zgyro": 10,
    "temperature": 14,
}


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Столбец по имени (без учёта регистра), иначе по номеру."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    if name in lower:
        key = lower[name]
    else:
        pos = _COLS[name]
        if pos > df.shape[1]:
            raise ValueError(f'В csv нет столбца "{name}" и нет столбца №{pos}.')
        key = df.columns[pos - 1]
    return pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)


def load_csv(cfg: Config) -> dict:
    """Возвращает dict: t [с], accX..Z [g], grsX..Z [°/с], temp [°C]."""
    c = cfg.csv
    df = pd.read_csv(cfg.file, header=0 if c.header else None, skipinitialspace=True)

    keep = np.ones(len(df), dtype=bool)
    if c.sys_id is not None:
        keep &= _col(df, "sys_id") == c.sys_id
    if c.comp_id is not None:
        keep &= _col(df, "comp_id") == c.comp_id
    df = df[keep]
    if df.empty:
        raise ValueError(f"В {cfg.file} нет строк после фильтра sys_id/comp_id.")

    d = {
        "t": _col(df, "time_boot_ms") * 1e-3,
        "accX":  _col(df, "xacc") * c.acc_scale,
        "accY": -_col(df, "zacc") * c.acc_scale,
        "accZ":  _col(df, "yacc") * c.acc_scale,
        "grsX":  _col(df, "xgyro") * c.gyro_scale,
        "grsY": -_col(df, "zgyro") * c.gyro_scale,
        "grsZ":  _col(df, "ygyro") * c.gyro_scale,
        "temp": _col(df, "temperature") * c.temp_scale,
    }
    bad = np.zeros(len(df), dtype=bool)
    for k in ("accX", "accY", "accZ", "grsX", "grsY", "grsZ"):
        bad |= np.isnan(d[k])
    if bad.any():
        print(f"ВНИМАНИЕ: удалено {bad.sum()} строк с пропусками; индексы считаются после удаления.")
        d = {k: v[~bad] for k, v in d.items()}
    return d


# ===================== ИНТЕРВАЛ ОБРАБОТКИ =====================


def resolve_range(d: dict, r: RangeCfg) -> tuple[int, int]:
    """Границы cfg.range -> номера строк (i0, i1), 1-based включительно."""
    n = len(d["accX"])
    if r.unit == "index":
        i0, i1 = r.start, r.stop
    elif r.unit == "sec":
        ts = d["t"] - d["t"][0]
        idx0 = np.nonzero(ts >= r.start)[0]
        idx1 = np.arange(n) if math.isinf(r.stop) else np.nonzero(ts <= r.stop)[0]
        if len(idx0) == 0 or len(idx1) == 0:
            raise ValueError(f"Интервал [{r.start} {r.stop}] с вне записи (0..{ts[-1]:.1f} с).")
        i0, i1 = idx0[0] + 1, idx1[-1] + 1
    else:
        raise ValueError('range.unit должен быть "index" или "sec".')
    i0 = max(1, int(round(i0)))
    i1 = n if math.isinf(i1) else min(n, int(round(i1)))
    if i1 - i0 < 10:
        raise ValueError(f"Слишком короткий интервал обработки: {i0}..{i1}.")
    return i0, i1


def pick_range(d: dict, unit: str) -> tuple[int, int]:
    """Выбор начала и конца мышью на обзорном графике (два клика)."""
    import matplotlib.pyplot as plt

    n = len(d["accX"])
    use_sec = unit == "sec"
    x = d["t"] - d["t"][0] if use_sec else np.arange(1, n + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, num="Выбор интервала обработки")
    for k in ("accX", "accY", "accZ"):
        ax1.plot(x, d[k], label=k)
    for k in ("grsX", "grsY", "grsZ"):
        ax2.plot(x, d[k], label=k)
    ax1.set_ylabel("acc, g"); ax2.set_ylabel("gyro, °/с")
    ax2.set_xlabel("время, с" if use_sec else "номер строки")
    ax1.set_title("Два клика: начало и конец интервала обработки")
    for a in (ax1, ax2):
        a.grid(True); a.legend(loc="upper right")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) < 2:
        raise RuntimeError("Не выбраны две точки.")
    p0, p1 = sorted(p[0] for p in pts)
    if use_sec:
        i0 = int(np.nonzero(x >= p0)[0][0]) + 1
        i1 = int(np.nonzero(x <= p1)[0][-1]) + 1
    else:
        i0, i1 = int(round(p0)), int(round(p1))
    i0, i1 = max(1, i0), min(n, i1)
    if i1 - i0 < 10:
        raise ValueError("Выбран слишком короткий интервал.")
    return i0, i1


# ===================== ДЕТЕКТОР ПОЛОК =====================


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _mov(x: np.ndarray, win: int, fn: str) -> np.ndarray:
    """Скользящая функция по столбцам, окно по центру, на краях окно сжимается (как movmedian в MATLAB)."""
    r = pd.DataFrame(x).rolling(window=win, center=True, min_periods=1)
    return getattr(r, fn)().to_numpy()


def snap_to_nominal(m: np.ndarray, o: PlateauCfg) -> tuple[np.ndarray, float]:
    """Привязка среднего вектора к эталону: по списку векторов или по осям."""
    if o.nominal is not None:
        nom = np.asarray(o.nominal, dtype=float)
        dist = np.linalg.norm(nom - m, axis=1)
        k = int(np.argmin(dist))
        return nom[k].copy(), float(dist[k])
    refv = np.zeros(3)
    dev = np.zeros(3)
    for c in range(3):
        ix = int(np.argmin(np.abs(o.vals - m[c])))
        dev[c] = abs(o.vals[ix] - m[c])
        refv[c] = o.vals[ix]
    return refv, float(dev.max())


def detect_plateaus(S: np.ndarray, o: PlateauCfg, ranges_local: list) -> tuple[np.ndarray, list]:
    """Совместный поиск полок по трём осям датчика.

    Полка = непрерывный участок, где локальный размах сглаженного сигнала <= tol
    одновременно по X, Y и Z. Среднее по полке привязывается к эталону ВЕКТОРОМ,
    поэтому оси не могут получить несогласованные границы и референсы.
    Полки, непохожие на эталон (рампы, переходы), отбраковываются, а не притягиваются.

    ranges_local — окна поиска [нач, конец] (1-based, включительно) в индексах обрезанных данных.
    Возвращает ref (N×3) и список полок; индексы a, b у полок 0-based локальные, включительно.
    """
    N = S.shape[0]
    ref = np.zeros((N, 3))
    segs = []
    for r0, r1 in ranges_local:
        ts = max(0, int(r0) - 1)
        te = N - 1 if math.isinf(r1) else min(N - 1, int(r1) - 1)
        if te - ts + 1 < o.min_len:
            continue
        X = S[ts:te + 1]
        n = X.shape[0]
        Sm = _mov(X, o.med, "median")
        W = min(o.win, n)
        lr = _mov(Sm, W, "max") - _mov(Sm, W, "min")
        stable = np.all(lr <= o.tol, axis=1)

        d = np.diff(np.concatenate([[0], stable.astype(int), [0]]))
        st = np.nonzero(d == 1)[0]
        en = np.nonzero(d == -1)[0] - 1

        for s0, e0 in zip(st, en):
            L = e0 - s0 + 1
            if L < o.min_len:
                continue
            t = max(o.trim_min, _round_half_up(o.trim_frac * L))
            a, b = s0 + t, e0 - t
            seg = dict(a=a + ts, b=b + ts, len=b - a + 1,
                       m=np.full(3, np.nan), sd=np.full(3, np.nan), drift=np.full(3, np.nan),
                       ref=np.full(3, np.nan), dist=np.nan, status="short")
            if seg["len"] >= 10:
                w = X[a:b + 1]
                h = w.shape[0] // 2
                m = w.mean(axis=0)
                seg["m"] = m
                seg["sd"] = w.std(axis=0, ddof=1)
                seg["drift"] = np.abs(w[:h].mean(axis=0) - w[-h:].mean(axis=0))
                refv, dist = snap_to_nominal(m, o)
                seg["ref"], seg["dist"] = refv, dist
                if np.any(seg["drift"] > o.drift_tol):
                    seg["status"] = "drift"
                elif dist > o.snap_tol:
                    seg["status"] = "far"
                elif math.isfinite(o.norm_tol) and abs(np.linalg.norm(m) - 1) > o.norm_tol:
                    seg["status"] = "norm"
                else:
                    seg["status"] = "ok"
                    ref[seg["a"]:seg["b"] + 1] = refv
            segs.append(seg)
    return ref, segs


def print_segments(segs: list, tag: str, unit: str, off: int) -> None:
    """Таблица полок; номера строк приведены к индексам полной записи (1-based)."""
    print(f"\n=== Полки {tag} ({unit}) ===")
    print(f"{'#':>3} {'нач':>7} {'кон':>7} {'длина':>5} | {'mX':>9} {'mY':>9} {'mZ':>9} | "
          f"{'sdX':>7} {'sdY':>7} {'sdZ':>7} | {'rX':>6} {'rY':>6} {'rZ':>6} | {'dist':>6} статус")
    for i, s in enumerate(segs, 1):
        print(f"{i:3d} {s['a'] + off + 1:7d} {s['b'] + off + 1:7d} {s['len']:5d} | "
              f"{s['m'][0]:9.3f} {s['m'][1]:9.3f} {s['m'][2]:9.3f} | "
              f"{s['sd'][0]:7.3f} {s['sd'][1]:7.3f} {s['sd'][2]:7.3f} | "
              f"{s['ref'][0]:6.2f} {s['ref'][1]:6.2f} {s['ref'][2]:6.2f} | {s['dist']:6.3f} {s['status']}")
    n_ok = sum(s["status"] == "ok" for s in segs)
    print(f"принято {n_ok} из {len(segs)}")


def plot_segments(S: np.ndarray, ref: np.ndarray, segs: list, tag: str, off: int):
    """Сигнал, референс и границы полок (принятые — зелёные, отбракованные — красные)."""
    import matplotlib.pyplot as plt

    x = off + np.arange(1, S.shape[0] + 1)
    fig, axes = plt.subplots(3, 1, sharex=True, num=f"Полки {tag}")
    for c, ax in enumerate(axes):
        ax.plot(x, S[:, c], "b")
        ax.plot(x, ref[:, c], "r--", lw=1.5)
        ax.grid(True)
        for s in segs:
            col = (0, 0.6, 0) if s["status"] == "ok" else (0.8, 0, 0)
            ax.axvspan(s["a"] + off + 1, s["b"] + off + 1, color=col, alpha=0.12, lw=0)
        ax.set_ylabel(f"{tag} {'XYZ'[c]}")
    axes[-1].set_xlabel("номер строки")
    return fig


# ===================== МНК =====================


def find_reference_intervals(r: np.ndarray) -> np.ndarray:
    """Участки, где референс ≠ 0 (смена значения референса тоже разделяет участки).
    Строки [начало, конец], 0-based, включительно."""
    r = np.asarray(r).ravel()
    ch = np.concatenate([[True], np.diff(r) != 0, [True]])
    nz = r != 0
    starts = np.nonzero(ch[:-1] & nz)[0]
    ends = np.nonzero(ch[1:] & nz)[0]
    return np.column_stack([starts, ends]) if len(starts) else np.zeros((0, 2), dtype=int)


def get_mean_values(aX, aY, aZ, rX, rY, rZ, axis: str) -> np.ndarray:
    """Средние на участках, где референс выбранной оси ≠ 0.
    Столбцы: 0 — средний референс оси, 1:4 — измерение XYZ, 4:7 — референс XYZ."""
    r = {"X": rX, "Y": rY, "Z": rZ}[axis]
    iv = find_reference_intervals(r)
    A = np.zeros((len(iv), 7))
    for i, (s, e) in enumerate(iv):
        sl = slice(s, e + 1)
        A[i] = [r[sl].mean(), aX[sl].mean(), aY[sl].mean(), aZ[sl].mean(),
                rX[sl].mean(), rY[sl].mean(), rZ[sl].mean()]
    return A


def _rows(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1, 0, 0, x, 0, 0, y, z, 0, 0, 0, 0],
                     [0, 1, 0, 0, y, 0, 0, 0, x, z, 0, 0],
                     [0, 0, 1, 0, 0, z, 0, 0, 0, 0, x, y]], dtype=float)


def mnk_pair(mX: np.ndarray, mY: np.ndarray, mZ: np.ndarray) -> np.ndarray:
    """МНК для одной пары полок по каждой оси. m*: 2×7."""
    blocks = (mX, mY, mZ)
    Z = np.concatenate([blocks[b][p, 1:4] - blocks[b][p, 4:7] for b in range(3) for p in range(2)])
    H = np.vstack([_rows(*blk[p, 4:7]) for blk in blocks for p in range(2)])
    return np.linalg.pinv(H.T @ H) @ H.T @ Z


def mnk(m_acc: list, m_grs: list) -> tuple[np.ndarray, np.ndarray]:
    """m_acc, m_grs — списки из трёх массивов (оси X, Y, Z). Оценки по парам полок: (n_pairs × 12)."""
    gX, gY, gZ = m_grs

    def run(mx, my, mz):
        n = min(mx.shape[0], my.shape[0], mz.shape[0])
        out = [mnk_pair(mx[i:i + 2], my[i:i + 2], mz[i:i + 2]) for i in range(0, n - 1, 2)]
        return np.array(out).reshape(-1, 12)

    return run(gX, gY, gZ), run(*m_acc)


def calibration(d: dict) -> tuple[np.ndarray, np.ndarray]:
    m_acc = [get_mean_values(d["accX"], d["accY"], d["accZ"],
                             d["ref_accX"], d["ref_accY"], d["ref_accZ"], ax) for ax in "XYZ"]
    m_grs = [get_mean_values(d["grsX"], d["grsY"], d["grsZ"],
                             d["ref_grsX"], d["ref_grsY"], d["ref_grsZ"], ax) for ax in "XYZ"]
    return mnk(m_acc, m_grs)


def mean_over_pairs(X: np.ndarray) -> np.ndarray:
    return np.zeros(12) if X.size == 0 else X.mean(axis=0)


# ===================== ГЛАВНАЯ ФУНКЦИЯ =====================


def run(cfg: Config) -> dict:
    full = load_csv(cfg)
    n = len(full["accX"])
    med_norm = float(np.median(np.linalg.norm(np.column_stack([full["accX"], full["accY"], full["accZ"]]), axis=1)))
    print(f"Загружено {n} отсчётов, медиана |acc| = {med_norm:.3f} g (ожидается около 1 при верных единицах)")

    # интервал обработки
    i0, i1 = pick_range(full, cfg.range.unit) if cfg.range.pick else resolve_range(full, cfg.range)
    print(f"Интервал обработки: строки {i0}..{i1} из {n} "
          f"({full['t'][i0 - 1] - full['t'][0]:.1f}..{full['t'][i1 - 1] - full['t'][0]:.1f} с от начала)")
    dat = {k: v[i0 - 1:i1] for k, v in full.items()}
    off = i0 - 1  # локальный индекс (0-based) + off = индекс в полной записи (0-based)

    def shift(ranges):  # окна из индексов полной записи в индексы обрезанных данных (1-based)
        return [[a - off, b - off] for a, b in ranges]

    accS = np.column_stack([dat["accX"], dat["accY"], dat["accZ"]])
    grsS = np.column_stack([dat["grsX"], dat["grsY"], dat["grsZ"]])
    refA, segA = detect_plateaus(accS, cfg.acc, shift(cfg.acc.ranges))
    refG, segG = detect_plateaus(grsS, cfg.grs, shift(cfg.grs.ranges))

    for j, ax in enumerate("XYZ"):
        dat[f"ref_acc{ax}"] = refA[:, j]
        dat[f"ref_grs{ax}"] = refG[:, j]

    print_segments(segA, "АКС", "g", off)
    print_segments(segG, "ГИРО", "°/с", off)

    X_grs_pairs, X_acc_pairs = calibration(dat)
    KK_grs, KK_acc = mean_over_pairs(X_grs_pairs), mean_over_pairs(X_acc_pairs)

    np.set_printoptions(precision=5, suppress=True, linewidth=140)
    print(f"Гироскопы: оценок по парам = {X_grs_pairs.shape[0]}\n{KK_grs}")
    print(f"Акселерометры: оценок по парам = {X_acc_pairs.shape[0]}\n{KK_acc}")

    if cfg.plot_seg:
        import matplotlib.pyplot as plt
        plot_segments(accS, refA, segA, "АКС", off)
        plot_segments(grsS, refG, segG, "ГИРО", off)
        plt.show()

    return dict(KK_grs=KK_grs, KK_acc=KK_acc, X_grs_pairs=X_grs_pairs, X_acc_pairs=X_acc_pairs,
                segments_acc=segA, segments_grs=segG, interval=(i0, i1))


# ===================== КОМАНДНАЯ СТРОКА =====================


def parse_args() -> tuple[Config, str | None]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="входной .csv")
    p.add_argument("--no-header", action="store_true", help="в файле нет строки заголовка")
    p.add_argument("--sys-id", type=int); p.add_argument("--comp-id", type=int)
    p.add_argument("--acc-scale", type=float); p.add_argument("--gyro-scale", type=float)
    p.add_argument("--temp-scale", type=float)
    p.add_argument("--unit", choices=["index", "sec"], help="единицы границ интервала обработки")
    p.add_argument("--start", type=float); p.add_argument("--stop", type=float)
    p.add_argument("--pick", action="store_true", help="выбрать начало и конец мышью")
    p.add_argument("--acc-range", type=float, nargs=2, metavar=("A", "B"),
                   help="окно поиска полок акс, индексы полной записи")
    p.add_argument("--grs-range", type=float, nargs=2, metavar=("A", "B"),
                   help="окно поиска полок гиро, индексы полной записи")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save", metavar="JSON", help="сохранить итоговые оценки в json")
    a = p.parse_args()

    cfg = Config()
    if a.file: cfg.file = a.file
    if a.no_header: cfg.csv.header = False
    if a.sys_id is not None: cfg.csv.sys_id = a.sys_id
    if a.comp_id is not None: cfg.csv.comp_id = a.comp_id
    if a.acc_scale is not None: cfg.csv.acc_scale = a.acc_scale
    if a.gyro_scale is not None: cfg.csv.gyro_scale = a.gyro_scale
    if a.temp_scale is not None: cfg.csv.temp_scale = a.temp_scale
    if a.unit: cfg.range.unit = a.unit
    if a.start is not None: cfg.range.start = a.start
    if a.stop is not None: cfg.range.stop = a.stop
    if a.pick: cfg.range.pick = True
    if a.acc_range: cfg.acc.ranges = [list(a.acc_range)]
    if a.grs_range: cfg.grs.ranges = [list(a.grs_range)]
    if a.no_plot: cfg.plot_seg = False
    return cfg, a.save


if __name__ == "__main__":
    config, save_path = parse_args()
    res = run(config)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"KK_grs": res["KK_grs"].tolist(), "KK_acc": res["KK_acc"].tolist(),
                       "interval": res["interval"]}, f, ensure_ascii=False, indent=2)
        print(f"Сохранено: {save_path}")
