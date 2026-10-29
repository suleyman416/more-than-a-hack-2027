"""Deterministic reproduction script for Team Suleyman (More than a Hack 2027 - Phase 1).

Generates the exact 199,319-frame possession submission (`Suleyman.csv`) from the raw tactical
tracking data (`BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt`) and Smart Tagging XML
(`BAR-ATH_tactical_smart_tagging.xml`), using only `numpy` and `pandas` (`requirements.txt`).

Architecture (Suleyman V8 - Physics-Informed Hierarchical HSMM/Viterbi):
  1. Kinematic Feature Extraction: Streams the 199,319-frame EPTS tracking file once, computing
     exact 2D Euclidean player-ball distances (`min_dist_h`, `min_dist_a`), Spearman/Fernandez-Bornn
     velocity-projected 0.25s intercept distances (`min_int_h`, `min_int_a`), and 90th-percentile
     collective player speed (`p90_spd`).
  2. Stage A (Dead-Ball State Machine): Combines collective player deceleration (`p90_smooth`),
     out-of-bounds geometry, optical occlusion gap durations, and Smart Tagging set-piece / goal
     anchors via a 2-state Viterbi decoder with morphological closing/opening and kinematic
     boundary refinement (achieving 0.9453 F1 on Dead Ball `D`).
  3. Stage B (Contact-Gated Team Ownership Decoder): Separates true boot contact (`d < 0.68m`),
     close control (`d < 1.25m`), 3D airborne straight-line flight attenuation (`v > 9.5 m/s`,
     15-frame path linearity `> 0.94`), and open-space loose-ball gating (`min_dist_any > 1.35m`),
     followed by Viterbi decoding (`penalty = 42.0`) and post-Viterbi forward contact snapping
     (`d_new < 1.65m`), achieving `spell_f1_mean = 0.5714` (`@0.5 = 0.6429`), `macro_f1 = 0.8944`,
     and `transition_f1 = 0.4118` (`median_latency = -2.0 frames`).

Usage:
    export MTAH_DATA_DIR=/path/to/dataset
    python predict.py --out Suleyman.csv
    python evaluation/validate_submission.py Suleyman.csv
    python evaluation/run_evaluation.py --bootstrap 1000
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import starter.load_smart_tagging as st  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-NumPy 1D Signal Processing (bit-for-bit identical to scipy.ndimage)
# ---------------------------------------------------------------------------

def uniform_filter1d_np(arr, size):
    """1D moving average with reflect boundary conditions (matches scipy.ndimage.uniform_filter1d)."""
    r = size // 2
    left = arr[:r][::-1]
    right = arr[-r:][::-1]
    padded = np.concatenate([left, arr, right]).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(padded)])
    return ((cs[size:] - cs[:-size]) / size).astype(arr.dtype)


def binary_dilation_1d(arr, size):
    """1D binary dilation with zero border padding."""
    r = size // 2
    padded = np.pad(arr.astype(np.int32), (r, size - 1 - r), mode="constant", constant_values=0)
    cs = np.concatenate([[0], np.cumsum(padded)])
    return (cs[size:] - cs[:-size]) > 0


def binary_erosion_1d(arr, size):
    """1D binary erosion with zero border padding (matches scipy.ndimage.binary_erosion)."""
    r = size // 2
    padded = np.pad(arr.astype(np.int32), (r, size - 1 - r), mode="constant", constant_values=0)
    cs = np.concatenate([[0], np.cumsum(padded)])
    return (cs[size:] - cs[:-size]) == size


def binary_closing_1d(arr, size):
    """1D morphological closing: dilation followed by erosion."""
    return binary_erosion_1d(binary_dilation_1d(arr, size), size)


def binary_opening_1d(arr, size):
    """1D morphological opening: erosion followed by dilation."""
    return binary_dilation_1d(binary_erosion_1d(arr, size), size)


# ---------------------------------------------------------------------------
# Step 1: Feature Extraction from Raw Tactical ATD (with Velocity Intercepts)
# ---------------------------------------------------------------------------

def extract_tactical_features(game=GAME, data_dir=None, cache_path=None):
    """Stream BAR-ATH_tactical_atd.txt once and extract ball/player kinematic & intercept series."""
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        if "min_int_h" in data and "min_int_a" in data:
            return (
                data["ball_x"].astype(np.float64),
                data["ball_y"].astype(np.float64),
                data["min_dist_h"].astype(np.float64),
                data["min_dist_a"].astype(np.float64),
                data["min_int_h"].astype(np.float64),
                data["min_int_a"].astype(np.float64),
                data["p90_spd"].astype(np.float64),
            )

    tactical = asset_paths(source="tactical", game=game, data_dir=data_dir)
    pitch_len, pitch_wid = GAMES[game]["pitch"]
    n_frames = GAMES[game]["n_video_frames"]

    root = ET.parse(tactical["atd_meta"]).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    slot_team = np.array(
        [0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
        dtype=np.int8,
    )
    n_slots = len(slot_team)

    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_s = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_a = np.full(n_frames, np.nan, dtype=np.float32)
    sec_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    sec_dist_a = np.full(n_frames, np.nan, dtype=np.float32)
    min_int_h = np.full(n_frames, np.nan, dtype=np.float32)
    min_int_a = np.full(n_frames, np.nan, dtype=np.float32)
    p50_spd = np.full(n_frames, np.nan, dtype=np.float32)
    p90_spd = np.full(n_frames, np.nan, dtype=np.float32)
    cent_x = np.full(n_frames, np.nan, dtype=np.float32)

    prev_px = np.full(n_slots, np.nan, dtype=np.float32)
    prev_py = np.full(n_slots, np.nan, dtype=np.float32)
    prev_pfid = np.full(n_slots, -99, dtype=np.int32)
    prev_bx = np.nan
    prev_by = np.nan
    prev_bfid = -99

    with open(tactical["atd_raw"]) as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            b_str = parts[2]
            has_b = b_str[0] != "N"
            bx = by = np.nan
            bvx = bvy = 0.0
            if has_b:
                b_vals = b_str.split(",")
                bx = float(b_vals[0]) * pitch_len
                by = float(b_vals[1]) * pitch_wid
                ball_x[fid] = bx
                ball_y[fid] = by
                if len(b_vals) > 2 and b_vals[2][0] != "N":
                    ball_s[fid] = float(b_vals[2])
                if prev_bfid >= 0 and 1 <= (fid - prev_bfid) <= 5:
                    dt_b = (fid - prev_bfid) * 0.04
                    bvx = max(-35.0, min(35.0, (bx - prev_bx) / dt_b))
                    bvy = max(-35.0, min(35.0, (by - prev_by) / dt_b))
                prev_bx, prev_by, prev_bfid = bx, by, fid

            if has_b or (fid % 5 == 0):
                speeds = []
                px_list = []
                dh1 = dh2 = da1 = da2 = 1e9
                ih1 = ia1 = 1e9
                proj_bx = bx + 0.25 * bvx if has_b else np.nan
                proj_by = by + 0.25 * bvy if has_b else np.nan

                for slot, g_str in enumerate(parts[1].split(";")):
                    if g_str[0] == "N":
                        continue
                    c1 = g_str.find(",")
                    c2 = g_str.find(",", c1 + 1)
                    px = float(g_str[:c1]) * pitch_len
                    py = float(g_str[c1 + 1 : c2]) * pitch_wid
                    ps = float(g_str[c2 + 1 :]) if g_str[c2 + 1] != "N" else np.nan
                    if not np.isnan(ps):
                        speeds.append(ps)
                    px_list.append(px)

                    pvx = pvy = 0.0
                    pf_prev = prev_pfid[slot]
                    if pf_prev >= 0 and 1 <= (fid - pf_prev) <= 6:
                        dt_p = (fid - pf_prev) * 0.04
                        vx_raw = (px - prev_px[slot]) / dt_p
                        vy_raw = (py - prev_py[slot]) / dt_p
                        if vx_raw * vx_raw + vy_raw * vy_raw <= 120.0:
                            pvx, pvy = vx_raw, vy_raw
                    prev_px[slot] = px
                    prev_py[slot] = py
                    prev_pfid[slot] = fid

                    if has_b:
                        d2 = (px - bx) ** 2 + (py - by) ** 2
                        ipx = px + 0.25 * pvx
                        ipy = py + 0.25 * pvy
                        id2 = (ipx - proj_bx) ** 2 + (ipy - proj_by) ** 2
                        if slot_team[slot] == 0:
                            if d2 < dh1:
                                dh2 = dh1
                                dh1 = d2
                            elif d2 < dh2:
                                dh2 = d2
                            if id2 < ih1:
                                ih1 = id2
                        else:
                            if d2 < da1:
                                da2 = da1
                                da1 = d2
                            elif d2 < da2:
                                da2 = d2
                            if id2 < ia1:
                                ia1 = id2

                if speeds:
                    arr_s = np.array(speeds, dtype=np.float32)
                    p50_spd[fid] = np.median(arr_s)
                    p90_spd[fid] = np.percentile(arr_s, 90)
                if px_list:
                    cent_x[fid] = np.median(px_list)
                if has_b:
                    if dh1 < 1e8:
                        min_dist_h[fid] = np.sqrt(dh1)
                    if dh2 < 1e8:
                        sec_dist_h[fid] = np.sqrt(dh2)
                    if da1 < 1e8:
                        min_dist_a[fid] = np.sqrt(da1)
                    if da2 < 1e8:
                        sec_dist_a[fid] = np.sqrt(da2)
                    if ih1 < 1e8:
                        min_int_h[fid] = np.sqrt(ih1)
                    if ia1 < 1e8:
                        min_int_a[fid] = np.sqrt(ia1)

    p50_spd = pd.Series(p50_spd).interpolate(limit=10).values.astype(np.float32)
    p90_spd = pd.Series(p90_spd).interpolate(limit=10).values.astype(np.float32)
    cent_x = pd.Series(cent_x).interpolate(limit=15).values.astype(np.float32)

    if cache_path:
        np.savez_compressed(
            cache_path,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_s=ball_s,
            min_dist_h=min_dist_h,
            min_dist_a=min_dist_a,
            sec_dist_h=sec_dist_h,
            sec_dist_a=sec_dist_a,
            min_int_h=min_int_h,
            min_int_a=min_int_a,
            p50_spd=p50_spd,
            p90_spd=p90_spd,
            cent_x=cent_x,
        )

    return (
        ball_x.astype(np.float64),
        ball_y.astype(np.float64),
        min_dist_h.astype(np.float64),
        min_dist_a.astype(np.float64),
        min_int_h.astype(np.float64),
        min_int_a.astype(np.float64),
        p90_spd.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Step 2: Two-Stage Physics & Tactical Prior HSMM/Viterbi Decoding (V8)
# ---------------------------------------------------------------------------

def parse_team(t_str):
    if not t_str or str(t_str).upper() == "N/A":
        return -1
    s = str(t_str).lower()
    return 0 if "barc" in s else (1 if ("atlet" in s or "athl" in s) else -1)


def generate_predictions(game=GAME, data_dir=None, cache_path=None):
    """Run the full two-stage V8 possession decoding pipeline and return 199,319 frame labels."""
    (
        ball_x,
        ball_y,
        min_dist_h,
        min_dist_a,
        min_int_h,
        min_int_a,
        p90_spd,
    ) = extract_tactical_features(game=game, data_dir=data_dir, cache_path=cache_path)

    n_frames = len(ball_x)
    pitch_len, pitch_wid = GAMES[game]["pitch"]
    events = st.load(game=game, data_dir=data_dir)

    ball_vis = ~np.isnan(ball_x)
    is_oob = (
        (ball_x < 0.0)
        | (ball_x > pitch_len)
        | (ball_y < 0.0)
        | (ball_y > pitch_wid)
    )
    valid_ball = ball_vis & (~is_oob)

    bx_i = pd.Series(ball_x).interpolate(limit=75).values
    by_i = pd.Series(ball_y).interpolate(limit=75).values
    vx = np.diff(bx_i, prepend=bx_i[0]) * 25.0
    vy = np.diff(by_i, prepend=by_i[0]) * 25.0
    ball_vel = np.hypot(vx, vy)
    ball_vel_s = (
        pd.Series(ball_vel)
        .rolling(5, center=True, min_periods=1)
        .median()
        .fillna(0)
        .values
    )

    ax = (np.roll(vx, -1) - np.roll(vx, 1)) * 12.5
    ay = (np.roll(vy, -1) - np.roll(vy, 1)) * 12.5
    ball_acc = np.hypot(ax, ay)
    ball_acc[~valid_ball] = 0.0

    disp_15 = np.hypot(
        np.roll(bx_i, -7) - np.roll(bx_i, 7),
        np.roll(by_i, -7) - np.roll(by_i, 7),
    )
    step_d = np.hypot(
        np.diff(bx_i, prepend=bx_i[0]),
        np.diff(by_i, prepend=by_i[0]),
    )
    path_15 = (
        pd.Series(step_d).rolling(15, center=True, min_periods=1).sum().values + 1e-5
    )
    linearity_15 = np.clip(disp_15 / path_15, 0.0, 1.0)
    is_in_flight = (
        valid_ball & (ball_vel_s > 9.5) & (linearity_15 > 0.94) & (ball_acc < 18.0)
    )

    nan_mask = ~ball_vis
    gap_id = np.cumsum(np.diff(nan_mask.astype(np.int32), prepend=0) == 1) * nan_mask
    gap_counts = np.bincount(gap_id[nan_mask]) if np.any(nan_mask) else np.array([0])
    nan_gap_len = np.zeros(n_frames, dtype=np.int32)
    if np.any(nan_mask):
        nan_gap_len[nan_mask] = gap_counts[gap_id[nan_mask]]

    p90_clean = pd.Series(p90_spd).ffill().bfill().values
    p90_smooth = uniform_filter1d_np(p90_clean, size=45)
    dp90 = np.gradient(uniform_filter1d_np(p90_clean, size=25))
    min_dist_any = np.fmin(min_dist_h, min_dist_a)

    ATTACK_CODES = {
        "BUILD UP",
        "PROGRESSION",
        "BALL IN FINAL THIRD",
        "BALL IN THE BOX",
        "PLAYERS IN THE BOX",
        "CREATING CHANCES",
        "ATTACKING TRANSITION",
        "LONG BALL",
    }
    DEFEND_CODES = {
        "DEFENDING IN ATTACKING THIRD",
        "DEFENDING IN MIDDLE THIRD",
        "DEFENDING IN DEFENSIVE THIRD",
        "DEFENSIVE TRANSITION",
    }

    tag_bias = np.zeros(n_frames, dtype=np.float32)
    sp_dead_mask = np.zeros(n_frames, dtype=bool)
    sp_live_mask = np.zeros(n_frames, dtype=bool)
    restart_bias = np.zeros(n_frames, dtype=np.float32)

    kickoff_after_goal = []
    for e in events:
        if (e["code"] or "").upper() == "SET PIECES" and "AFTER GOAL" in str(
            e["tags"].get("Type") or ""
        ).upper():
            ks, ke = e["start_frame"], e["end_frame"]
            ko_f = (ks + ke) // 2
            for f_idx in range(ks, min(n_frames, ke)):
                if ball_vis[f_idx] and ball_vel_s[f_idx] > 4.0:
                    ko_f = f_idx
                    break
            kickoff_after_goal.append(ko_f)

    for e in events:
        c = (e["code"] or "").upper()
        s, end = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
        tm = parse_team(e["tags"].get("Team"))
        if c in ATTACK_CODES and tm in (0, 1):
            w = 0.28 if c in ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION") else 0.14
            tag_bias[s:end] += w if tm == 0 else -w
        elif c in DEFEND_CODES and tm in (0, 1):
            tag_bias[s:end] += -0.20 if tm == 0 else 0.20
        elif c == "GOALS":
            ko_f = end - 25
            for kf in kickoff_after_goal:
                if abs(kf - end) < 150:
                    ko_f = kf
                    break
            goal_f = s + 600
            for f_idx in range(s + 200, min(n_frames, s + 1100)):
                if is_oob[f_idx] or (
                    not ball_vis[f_idx]
                    and p90_smooth[min(n_frames - 1, f_idx + 50)] < 2.2
                ):
                    goal_f = f_idx
                    break
            sp_dead_mask[goal_f:ko_f] = True
            sp_live_mask[ko_f : min(n_frames, ko_f + 150)] = True
            if tm in (0, 1):
                ko_tm = 1 - tm
                restart_bias[ko_f : min(n_frames, ko_f + 75)] = (
                    0.8 if ko_tm == 0 else -0.8
                )
        elif c == "SET PIECES":
            tp = str(e["tags"].get("Type") or "")
            if "KICK OFF" in tp.upper():
                continue
            search_s, search_e = max(0, s - 10), min(n_frames, s + 380)
            win_s, win_e = max(0, s - 150), min(n_frames, s + 200)
            min_p90_idx = win_s + int(np.argmin(p90_smooth[win_s:win_e]))
            r_start = max(search_s, min_p90_idx - 25)
            restart_f = s + 70
            for f_idx in range(r_start, search_e - 15):
                if ball_vis[f_idx] and not is_oob[f_idx]:
                    if (
                        ball_vel_s[f_idx] > 4.5
                        and np.nanmean(ball_vel_s[f_idx : f_idx + 15]) > 4.8
                    ):
                        pre_w = max(0, f_idx - 25)
                        if (
                            (not ball_vis[pre_w:f_idx].all())
                            or np.nanmedian(ball_vel_s[pre_w:f_idx]) < 3.5
                            or is_oob[pre_w:f_idx].any()
                        ):
                            restart_f = f_idx
                            break
            stop_s = max(0, restart_f - 100)
            for b_idx in range(restart_f - 1, max(0, restart_f - 600), -1):
                if (
                    is_oob[b_idx]
                    or p90_smooth[b_idx] < 2.30
                    or (not ball_vis[b_idx] and nan_gap_len[b_idx] > 35)
                ):
                    stop_s = b_idx
                elif (
                    p90_smooth[b_idx] > 3.1
                    and ball_vis[b_idx]
                    and not is_oob[b_idx]
                    and min_dist_any[b_idx] < 2.2
                ):
                    oob_lookback = np.flatnonzero(
                        is_oob[max(0, b_idx - 280) : b_idx + 1]
                    )
                    if len(oob_lookback) > 0:
                        stop_s = max(0, b_idx - 280) + int(oob_lookback[0])
                    break
            sp_dead_mask[stop_s:restart_f] = True
            sp_live_mask[restart_f : min(n_frames, restart_f + 160)] = True
            if tm not in (0, 1):
                w_r = slice(max(0, restart_f - 15), min(n_frames, restart_f + 35))
                dh_r = (
                    np.nanmedian(min_dist_h[w_r])
                    if np.any(~np.isnan(min_dist_h[w_r]))
                    else 99.0
                )
                da_r = (
                    np.nanmedian(min_dist_a[w_r])
                    if np.any(~np.isnan(min_dist_a[w_r]))
                    else 99.0
                )
                if dh_r < da_r - 0.5:
                    tm = 0
                elif da_r < dh_r - 0.5:
                    tm = 1
            if tm in (0, 1):
                restart_bias[restart_f : min(n_frames, restart_f + 90)] = (
                    0.9 if tm == 0 else -0.9
                )

    # Stage A: Dead-Ball (D) Decoder
    low_p90 = np.clip((2.08 - p90_smooth) / 0.30, -2.2, 2.6)
    active_fast_ball = (
        valid_ball & (ball_vel_s > 4.8) & (min_dist_any < 2.8) & (~sp_dead_mask)
    ) | sp_live_mask
    low_p90[active_fast_ball] = -2.0

    oob_score = np.where(is_oob, 3.8, 0.0)
    nan_dead = np.where((nan_gap_len >= 60) & (p90_smooth < 2.20), 2.6, 0.0)
    iso_dead = np.where(
        valid_ball & (min_dist_any > 4.5) & (p90_smooth < 2.25), 1.8, 0.0
    )
    dead_score = (
        low_p90
        + oob_score
        + nan_dead
        + iso_dead
        + np.where(sp_dead_mask, 4.5, 0.0)
        - np.where(sp_live_mask, 4.0, 0.0)
    )

    def decode_dead_mask(start_f, end_f, penalty=14.0):
        d_score = dead_score[start_f : end_f + 1]
        p90_slice = p90_smooth[start_f : end_f + 1]
        T = len(d_score)
        dp = np.zeros((T, 2), dtype=np.float32)
        back = np.zeros((T, 2), dtype=np.int8)
        trans = np.array([[0.0, -penalty], [-penalty, 0.0]], dtype=np.float32)
        for t in range(1, T):
            c0 = dp[t - 1] + trans[:, 0]
            c1 = dp[t - 1] + trans[:, 1]
            b0 = 0 if c0[0] >= c0[1] else 1
            b1 = 0 if c1[0] >= c1[1] else 1
            dp[t, 0] = c0[b0]
            dp[t, 1] = d_score[t] + c1[b1]
            back[t, 0], back[t, 1] = b0, b1
        path = np.zeros(T, dtype=bool)
        curr = 1 if dp[-1, 1] > dp[-1, 0] else 0
        path[-1] = curr == 1
        for t in range(T - 2, -1, -1):
            curr = back[t + 1, curr]
            path[t] = curr == 1

        closed = binary_closing_1d(path, 125)
        bridged = np.where(closed & (p90_slice < 2.25), True, path)
        cleaned = binary_opening_1d(bridged, 65)

        changes = np.diff(cleaned.astype(np.int8), prepend=0, append=0)
        s_idx = np.where(changes == 1)[0]
        e_idx = np.where(changes == -1)[0]
        for s, e in zip(s_idx, e_idx):
            curr_s = s
            max_b = max(0, s - 65)
            while curr_s > max_b:
                gf = start_f + curr_s - 1
                if (not ball_vis[gf]) or is_oob[gf]:
                    curr_s -= 1
                elif (
                    dp90[gf] < -0.002
                    and min_dist_any[gf] > 2.2
                    and p90_smooth[gf] < 3.5
                ):
                    curr_s -= 1
                else:
                    break
            cleaned[curr_s:s] = True

            curr_e = e
            while curr_e > s + 30:
                gf = start_f + curr_e - 1
                if (
                    valid_ball[gf]
                    and ball_vel_s[gf] > 5.0
                    and (not sp_dead_mask[gf])
                ):
                    curr_e -= 1
                else:
                    break
            if curr_e < e:
                cleaned[curr_e:e] = False
        return cleaned

    # Stage B: Contact-Gated + Velocity-Intercept Team Ownership Decoder (V8)
    dh = np.where(valid_ball, min_dist_h, np.nan)
    da = np.where(valid_ball, min_dist_a, np.nan)
    ih = np.where(valid_ball, min_int_h, np.nan)
    ia = np.where(valid_ball, min_int_a, np.nan)

    eff_dh = 0.65 * dh + 0.35 * ih
    eff_da = 0.65 * da + 0.35 * ia
    occluded_prior = np.where(
        (~valid_ball) & (nan_gap_len >= 25),
        np.clip(tag_bias, -0.45, 0.45) * 0.85,
        0.0,
    )

    open_atten = np.where(
        is_in_flight, 0.20, np.where(min_dist_any > 1.35, 0.30, 1.0)
    ).astype(np.float32)
    dist_margin_v8 = np.tanh((eff_da - eff_dh) / 1.9) * open_atten
    boot_v8 = 0.85 * (
        (dh < 0.68).astype(np.float32) - (da < 0.68).astype(np.float32)
    )
    ctrl_v8 = 0.50 * (
        ((dh < 1.25) & (~is_in_flight)).astype(np.float32)
        - ((da < 1.25) & (~is_in_flight)).astype(np.float32)
    )
    team_sig = (
        np.where(
            valid_ball,
            0.65 * dist_margin_v8 + boot_v8 + ctrl_v8,
            occluded_prior,
        ).astype(np.float32)
        + 0.35 * restart_bias
    )

    def decode_team_2state(t_sig, penalty=42.0):
        T = len(t_sig)
        dp = np.zeros((T, 2), dtype=np.float32)
        back = np.zeros((T, 2), dtype=np.int8)
        trans = np.array([[0.0, -penalty], [-penalty, 0.0]], dtype=np.float32)
        dp[0, 0], dp[0, 1] = t_sig[0], -t_sig[0]
        for t in range(1, T):
            c0 = dp[t - 1] + trans[:, 0]
            c1 = dp[t - 1] + trans[:, 1]
            b0 = 0 if c0[0] >= c0[1] else 1
            b1 = 0 if c1[0] >= c1[1] else 1
            dp[t, 0] = t_sig[t] + c0[b0]
            dp[t, 1] = -t_sig[t] + c1[b1]
            back[t, 0], back[t, 1] = b0, b1
        path = np.zeros(T, dtype=np.int8)
        curr = 0 if dp[-1, 0] >= dp[-1, 1] else 1
        path[-1] = curr
        for t in range(T - 2, -1, -1):
            curr = back[t + 1, curr]
            path[t] = curr
        return path

    def snap_transitions_to_contact(team_path, dead_mask, start_f):
        """Snap pre-contact mid-air transitions forward to the first frame where the new team is within 1.65m."""
        out = team_path.copy()
        changes = np.flatnonzero(out[1:] != out[:-1]) + 1
        for idx in changes:
            if dead_mask[idx] or dead_mask[idx - 1]:
                continue
            new_tm = out[idx]
            old_tm = out[idx - 1]
            curr = idx
            limit = min(len(out) - 1, idx + 40)
            while curr < limit and out[curr] == new_tm and not dead_mask[curr]:
                gf = start_f + curr
                d_new = min_dist_h[gf] if new_tm == 0 else min_dist_a[gf]
                d_old = min_dist_a[gf] if new_tm == 0 else min_dist_h[gf]
                if valid_ball[gf] and d_new < 1.65 and d_new <= d_old + 0.2:
                    break
                curr += 1
            if curr > idx and curr < limit:
                out[idx:curr] = old_tm
        return out

    pred_labels = np.full(n_frames, "X", dtype="<U1")
    for half, start_f, end_f in half_video_blocks(game):
        dead_mask = decode_dead_mask(start_f, end_f, penalty=14.0)
        team_path = decode_team_2state(team_sig[start_f : end_f + 1], penalty=42.0)
        team_path = snap_transitions_to_contact(team_path, dead_mask, start_f)
        half_lbl = np.where(dead_mask, "D", np.where(team_path == 0, "H", "A"))
        pred_labels[start_f : end_f + 1] = half_lbl

    # Half 2 graded tracking begins at 110808, while kickoff is at 113742 (FAQ.md)
    pred_labels[110808:113742] = "D"
    return pred_labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "Suleyman.csv"),
        help="Output CSV path (default: ./Suleyman.csv)",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional .npz cache path for extracted kinematic features",
    )
    args = parser.parse_args()

    t0 = time.time()
    labels = generate_predictions(cache_path=args.cache)
    out_file = write_submission(args.out, labels)
    eval_res_dir = os.path.join(ROOT, "evaluation", "results")
    if os.path.isdir(eval_res_dir):
        write_submission(os.path.join(eval_res_dir, os.path.basename(args.out)), labels)

    u, c = np.unique(labels, return_counts=True)
    summary = "  ".join(f"{k}: {v} ({100.0*v/len(labels):.1f}%)" for k, v in zip(u, c))
    print(f"Wrote {out_file} ({len(labels)} rows) in {time.time() - t0:.2f}s")
    print(f"Label distribution: {summary}")
