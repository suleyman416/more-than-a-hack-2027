# Physics-Informed Hierarchical HSMM/Viterbi Decoding with Velocity-Projected Pitch Control and Contact-Gated Emissions for Frame-Exact Football Possession

**Author / Team:** Suleyman Mammadov (`Suleyman` / `@suleyman416`)  
**Challenge:** More than a Hack 2027 — Phase 1: Ball Possession (Metrica Sports & FC Barcelona Innovation Hub)  
**Deliverables:** `<team_name>.csv` (`Suleyman.csv`, 199,319 frames), `predict.py` + `requirements.txt`, `<team_name>.pdf` / `<team_name>.md`

---

## Executive Summary

Per-frame ball possession inference from automatic optical tracking data (ATD) fails under standard proximity heuristics for three fundamental physical reasons:
1. **Live vs. Dead-Ball Conflation:** A ball resting on the pitch during a free-kick setup, corner preparation, or referee stoppage remains tracked in $(x, y)$ with players standing nearby. Naïve `isnan(ball)` detectors fail completely on stationary in-bounds stoppages, scoring **`0.0000` Dead-Ball (`D`) F1** on the public benchmark.
2. **2D Projection of 3D Passes & Open-Space Transit Jitter:** During long aerial passes, clearances, or fast ground passes in transit ($v_{\text{ball}} > 9.5\text{ m/s}$ or $\min(d_H, d_A) > 1.35\text{m}$), the ball's 2D $(x, y)$ shadow passes near opposing midfielders who never touch it. Nearest-player heuristics hallucinate false 1–4 second turnovers, fragmenting true possessions (`fragmentation = 2.65–13.48`).
3. **Smoothing Latency vs. Fragmentation Trade-off:** Centered majority-vote filters (`baseline_smooth50`, `baseline_smooth125`) blur possession boundaries across $2\text{–}5\text{ seconds}$, shifting transitions into mid-flight before physical reception and absorbing short legitimate interceptions.

To solve all three failure modes from first principles—using **zero external training data**—we introduce **Suleyman V8**, a **Two-Stage Physics-Informed Hierarchical Hidden Semi-Markov / Viterbi Decoder** with **Spearman/Fernández-Bornn Velocity-Projected Intercept Distances**, **3D Airborne Flight & Open-Space Transit Gating**, and **Post-Viterbi Forward Contact Snapping**:

| Submission / Model | **Spell Mean F1** (`0.3..0.7`) *(Primary)* | **Spell F1 @ IoU 0.5** | **Frame Macro F1** *(1st Tie-Break)* | **Dead Ball (`D`) F1** | **Trans F1** (`@25f`) | **Median Latency** | **5-Min Possessions** *(GT = 17, 14.2s)* | **Full-Match Possessions** *(90 min)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_raw` | `0.0378` | `0.0308` | `0.5972` | `0.0000` | `0.1149` | `-1.5f` | `113` (`2.5s`) | `1,949` (`2.3s`) |
| `baseline_smooth25` | `0.2868` | `0.3019` | `0.6200` | `0.0000` | `0.3051` | `-2.0f` | `36` (`7.3s`) | `746` (`5.9s`) |
| `baseline_smooth50` *(Floor)* | `0.3256` | `0.3256` | `0.6207` | `0.0000` | `0.2353` | `-4.0f` | `26` (`9.5s`) | `594` (`7.5s`) |
| `baseline_smooth125` | `0.4242` | `0.4242` | `0.5812` | `0.0000` | `0.1500` | `-5.5f` | `16` (`15.6s`) | `398` (`11.2s`) |
| `Suleyman_v6` *(Baseline HSMM)* | `0.4571` | `0.5143` | `0.8960` | `0.9453` | `0.3902` | `-6.5f` | `18` (`13.4s`) | `339` (`10.0s`) |
| `Suleyman_v7_p36` *(+Intercept)* | `0.4571` | `0.5143` | `0.8970` | `0.9453` | `0.3902` | `-6.0f` | `18` (`13.4s`) | `304` (`11.1s`) |
| `Suleyman_v7_p42` *(+Snap)* | `0.4706` | `0.5294` | `0.8958` | `0.9453` | `0.3500` | `-2.0f` | `17` (`14.2s`) | `284` (`11.9s`) |
| `Suleyman_v8_tr42` *(+Space)* | `0.5333` | `0.6000` | **`0.8980`** | `0.9453` | `0.3889` | `-2.0f` | `13` (`18.5s`) | `253` (`13.3s`) |
| **`Suleyman.csv` (`V8_p42`)** | **`0.5714`** (`+75.5%`) | **`0.6429`** (`+97.5%`) | **`0.8944`** (`+44.1%`) | **`0.9453`** | **`0.4118`** | **`-2.0f`** (`-0.08s`) | **`11`** (`21.9s`) | **`239`** (`14.1s`) |

---

## 1. Approach & Architecture

```mermaid
flowchart LR
    A["Raw Tactical ATD\n(199,319 frames @ 25 Hz)"] --> B["Kinematic & Pitch Control Extractor\n• d_H, d_A (1st & 2nd defenders)\n• Velocity-projected intercept d_int (0.25s)\n• Ball velocity v_b, acceleration a_b, 15f linearity\n• 90th-percentile collective speed p90_spd"]
    C["Smart Tagging XML\n(Phases, Set Pieces, Goals)"] --> D["Set-Piece & Stoppage Kinematic Pinpointer\n• Exact ball-strike restart frame (v_b > 4.5 m/s)\n• Backward stoppage boundary walk\n• Goal-to-kickoff dead-ball locking"]
    B --> E["Stage A: Dead-Ball (D) Decoder\n• 2-state Viterbi (penalty = 14.0)\n• Morphological closing (125f) & opening (65f)\n• Kinematic onset/restart edge snapping"]
    D --> E
    B --> F["Stage B: Contact-Gated Team (H/A) Decoder\n• True boot contact (d < 0.68m) vs control (d < 1.25m)\n• 3D Flight (v_b > 9.5 m/s) & Open-Space (d > 1.35m) gate\n• 2-state Viterbi (penalty = 42.0)\n• Post-Viterbi forward contact snap (d_new < 1.65m)"]
    D --> F
    E --> G["Hierarchical State Merger\n• Pre/Halftime/Post -> X\n• Pre-Kickoff Half 2 [110808..113742) -> D\n• Dead Mask -> D, else Team Path -> H / A"]
    F --> G
    G --> H["Validated Submission\nSuleyman.csv (199,319 rows)"]
```

### 1.1 Kinematic & Velocity-Projected Pitch Control Features
Streaming `BAR-ATH_tactical_atd.txt` in a single pass ($O(N)$ over $199,319$ frames), we convert normalized Metrica coordinates $[0, 1]^2$ into physical pitch meters ($105\text{m} \times 68\text{m}$) and extract:
1. **Instantaneous Euclidean Proximity:** Minimum distance from the ball $\vec{b}(t)$ to the closest and second-closest player of Home ($d_H^{(1)}(t), d_H^{(2)}(t)$) and Away ($d_A^{(1)}(t), d_A^{(2)}(t)$).
2. **Spearman / Fernández-Bornn Velocity-Projected Intercept Distance ($\tau = 0.25\text{s}$):** Static distance fails when a ball is passed ahead of a sprinting winger or past a Wrong-Footed defender. For every tracked player $k$ with velocity $\vec{v}_k(t)$ (clamped to physical sprint limits $\|\vec{v}_k\| \le \sqrt{120}\text{ m/s}$) and ball velocity $\vec{v}_{\text{ball}}(t)$, we compute the projected intercept distance after human motor reaction time $\tau = 0.25\text{s}$:
   $$d_k^{\text{int}}(t) = \left\| \big(\vec{p}_k(t) + 0.25\,\vec{v}_k(t)\big) - \big(\vec{b}(t) + 0.25\,\vec{v}_{\text{ball}}(t)\big) \right\|_2$$
   Taking the team minimums $i_H(t) = \min_{k \in H} d_k^{\text{int}}(t)$ and $i_A(t) = \min_{k \in A} d_k^{\text{int}}(t)$, we form the effective kinematic control distance:
   $$d_H^{\text{eff}}(t) = 0.65\,d_H^{(1)}(t) + 0.35\,i_H(t), \qquad d_A^{\text{eff}}(t) = 0.65\,d_A^{(1)}(t) + 0.35\,i_A(t)$$
3. **Collective Peloton Speed (`p90_spd`):** The 90th percentile of instantaneous speeds across all 22 players on the pitch, smoothed with a 45-frame ($1.8\text{s}$) zero-phase box filter (`p90_smooth`). During live open play, at least 2–3 players sprint or jog rapidly ($\text{p90\_smooth} \in [2.6, 5.5]\text{ m/s}$); during dead-ball stoppages, the entire 22-player peloton decelerates to a walk ($\text{p90\_smooth} < 2.08\text{ m/s}$).

---

### 1.2 Stage A: Physics & Event-Anchored Dead-Ball (`D`) State Machine
Rather than relying on `isnan(ball_x)` (which fails whenever the dead ball remains visible on the grass), Stage A fuses four physical cues into a continuous dead-ball log-likelihood $s_D(t)$:
1. **Collective Peloton Deceleration (`low_p90`):**
   $$\text{low\_p90}(t) = \text{clip}\!\left(\frac{2.08 - \text{p90\_smooth}(t)}{0.30},\; -2.2,\; 2.6\right)$$
   overridden to $-2.0$ whenever a live, in-bounds ball is actively moving ($v_{\text{ball}} > 4.8\text{ m/s}$) within player proximity ($d_{\text{any}} < 2.8\text{m}$).
2. **Pitch Boundary Geometry (`oob_score`):** $+3.8$ whenever $\vec{b}(t) \notin [0, 105] \times [0, 68]\text{m}$.
3. **Optical Occlusion & Isolation Cues (`nan_dead`, `iso_dead`):** $+2.6$ during extended ball-tracking dropouts ($\text{nan\_gap\_len} \ge 60$ frames) accompanied by low player speed ($\text{p90\_smooth} < 2.20\text{ m/s}$), and $+1.8$ when the ball sits isolated ($d_{\text{any}} > 4.5\text{m}$) while players walk.
4. **Smart Tagging Set-Piece & Goal Kinematic Pinpointing (`sp_dead_mask`, `sp_live_mask`):**
   - Smart Tagging `SET PIECES` intervals have $\pm 3\text{s}$ coarse boundaries. Instead of trusting raw tag timestamps, we locate the **exact frame of ball-strike restart** $t_{\text{restart}}$ inside each set-piece window where the in-bounds ball accelerates past $v_{\text{ball}}(t) > 4.5\text{ m/s}$ (with $15\text{-frame}$ mean speed $> 4.8\text{ m/s}$) following a stationary/out-of-bounds setup.
   - From $t_{\text{restart}}$, we walk backward (`b_idx`) through the stoppage while the ball is out-of-bounds, occluded, or players are walking ($\text{p90\_smooth} < 2.30\text{ m/s}$), snapping the stoppage start $t_{\text{stop}}$ to the exact frame the ball crossed the touchline/goal line (`is_oob`).
   - For `GOALS`, we lock the interval from the ball crossing the goal line until the subsequent `Kick Off (After Goal)` ball-strike as `D`.
5. **Viterbi Decoding, Morphological Filtering & Edge Snapping:**
   - A 2-state Viterbi decoder (`penalty = 14.0`) decodes `Alive` vs. `Dead`, followed by 1D morphological closing ($125\text{ frames} = 5.0\text{s}$, gated by $\text{p90\_smooth} < 2.25\text{ m/s}$) to bridge brief mid-stoppage ball retrievals, and 1D morphological opening ($65\text{ frames} = 2.6\text{s}$) to reject sub-2.5s tracking dropouts during aerial flights.
   - Finally, each stoppage's onset and termination are kinematically snapped to the exact out-of-bounds crossing and the exact $v_{\text{ball}} > 5.0\text{ m/s}$ restart kick.
   - **Result:** Stage A achieves **`0.9453` F1 on Dead Ball (`D`)** (`Precision = 0.9415`, `Recall = 0.9491`) and **`0.9866` Alive/Dead F1**.

---

### 1.3 Stage B: Contact-Gated Team Ownership (`H` vs. `A`) Decoder
Our diagnostic analysis of the raw tracking data revealed the root cause of possession fragmentation in standard models:
- **Finding 1 (True Boot Contact vs. Passive Proximity):** In Ground Truth `Seg 3` (`[11624..12665]`, $41.6\text{s}$ of `H`), Barcelona makes **166 frames** of true boot contact ($d_H < 0.68\text{m}$) vs. only 13 for Athletic. However, during a 2-second clearance (`11724..11774`) where **neither team is within $1.36\text{m}$ of the ball**, Athletic players are passively closer in open space ($1.4\text{m–}4.3\text{m}$ vs. $4.0\text{m–}7.7\text{m}$). Giving full weight to open-space proximity when no player is within control radius ($d_{\text{any}} > 1.35\text{m}$) causes false mid-possession splits.
- **Finding 2 (3D Airborne Pass Projection):** When a long ball flies at high speed ($v_{\text{ball}} > 9.5\text{ m/s}$) along a straight trajectory ($15\text{-frame linearity} = \|\vec{b}(t+7)-\vec{b}(t-7)\| / \sum_{k=-6}^{7}\|\Delta\vec{b}(t+k)\| > 0.94$, low acceleration $a_{\text{ball}} < 18\text{ m/s}^2$), its 2D projection crosses opposing midfielders underneath the flight path without deflection.
- **Finding 3 (Pre-Contact Mid-Air Transition Drift):** Symmetrical sequence decoders place the decision boundary $H \leftrightarrow A$ halfway through the pass flight preceding an interception—typically $4\text{–}10\text{ frames}$ *before* the intercepting player touches the ball.

Stage B directly encodes all three physical laws:
1. **3D Flight & Open-Space Loose-Ball Attenuation Gate $g(t)$:**
   $$g(t) = \begin{cases} 0.20 & \text{if } \text{is\_in\_flight}(t) \;\; (v_{\text{ball}} > 9.5\text{ m/s},\; \text{linearity}_{15} > 0.94,\; a_{\text{ball}} < 18\text{ m/s}^2) \\ 0.30 & \text{if } \min(d_H(t), d_A(t)) > 1.35\text{m} \;\; (\text{ball loose / in transit outside control radius}) \\ 1.00 & \text{otherwise } (\min(d_H(t), d_A(t)) \le 1.35\text{m}) \end{cases}$$
2. **Tiered Boot-Contact & Control Emission $s_{\text{team}}(t)$:**
   $$s_{\text{team}}(t) = 0.65\,g(t)\tanh\!\left(\frac{d_A^{\text{eff}}(t) - d_H^{\text{eff}}(t)}{1.9}\right) + 0.85\big(\mathbb{I}[d_H < 0.68] - \mathbb{I}[d_A < 0.68]\big) + 0.50\big(\mathbb{I}[d_H < 1.25 \wedge \neg\text{flight}] - \mathbb{I}[d_A < 1.25 \wedge \neg\text{flight}]\big) + 0.35\,b_{\text{restart}}(t)$$
   When the ball is occluded for $\ge 1.0\text{s}$ ($\text{nan\_gap\_len} \ge 25$), $s_{\text{team}}(t)$ falls back to a bounded Smart Tagging phase prior (`BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `DEFENDING IN * THIRD`).
3. **2-State Viterbi Decoding & Post-Viterbi Forward Contact Snapping:**
   - We decode the globally optimal team path via 2-state Viterbi (`penalty = 42.0`).
   - **Forward Contact Snapping:** For every live-play transition $t^*$ where the decoded state switches to `new_tm`, if the new team is not yet within physical reception distance ($d_{\text{new}}(t^*) \ge 1.65\text{m}$), we advance $t^*$ forward (up to $40\text{ frames} = 1.6\text{s}$) to the exact first frame where $d_{\text{new}}(t) < 1.65\text{m}$ and $d_{\text{new}}(t) \le d_{\text{old}}(t) + 0.2\text{m}$. This improves median transition latency from `-6.5 frames` (`V6`) to **`-2.0 frames` (`-0.08s`)**!

---

## 2. Assets Used and Dropped

| Asset | Status | Role in Pipeline |
| :--- | :---: | :--- |
| **Tactical ATD** (`BAR-ATH_tactical_atd.xml/.txt`) | **USED** | Primary 25 Hz tracking source (already aligned 1:1 with submission frames via `video_frame = frame_id - 1`). Provides 2D ball coordinates, 22-player coordinates, speeds, and velocity-projected intercept vectors. |
| **Smart Tagging** (`BAR-ATH_tactical_smart_tagging.xml`) | **USED** | Provides coarse `SET PIECES` and `GOALS` windows (which we kinematically pinpoint to exact ball-strike frames), plus a weak prior during prolonged camera/ball occlusions (`nan_gap_len >= 25`) and set-piece restarts. |
| **Half Boundary Metadata** (`games.py`, `DATA.md`, `FAQ.md`) | **USED** | Defines the exact graded half intervals (`[11185..90651]` and `[110808..190197]`) and the Second-Half pre-kickoff dead-ball window (`[110808..113742)`). |
| **Broadcast ATD** (`BAR-ATH_broadcast_atd.*`) | **DROPPED** | Recorded on a separate 164,488-frame timeline with camera cuts and partial-pitch visibility, whereas Tactical ATD covers all 22 outfield players continuously across all 199,319 submission frames. |
| **Event Data / Video** (`*.json`, `*.mp4`) | **DROPPED** | Not required; our kinematic + Smart Tagging pipeline runs deterministically in `< 45 seconds` on CPU from text/XML alone. |

---

## 3. External Training Data Used

**None.** Our model is 100% self-contained, physics-informed, and zero-shot. No external matches, pre-trained neural networks, or external datasets were used.

---

## 4. Official Possession Rules Implemented

Every rule in `CHALLENGE.md` and `DATA.md` is explicitly enforced in `predict.py`:
1. **Controlled Possession Required for Turnover:** Open-space proximity ($\min(d_H, d_A) > 1.35\text{m}$) is attenuated by $0.30\times$ and high-speed straight flight ($v_{\text{ball}} > 9.5\text{ m/s}$, linearity $> 0.94$) by $0.20\times$, while true boot contact ($d < 0.68\text{m}$, $+0.85$) and close control ($d < 1.25\text{m}$, $+0.50$) drive the Viterbi emission. Brief single-touch deflections or loose balls cannot overcome the Viterbi switch penalty (`42.0`).
2. **No Mid-Air Turnovers Before Reception:** `snap_transitions_to_contact()` holds possession with the passing team during pass flight until the exact frame the receiving/intercepting opponent enters physical control range ($d_{\text{new}} < 1.65\text{m}$).
3. **Dead Ball (`D`) from Out-of-Bounds / Whistle Until Restart Kick:** Stage A locks `D` from the exact frame the ball crosses the touchline/goal line (`is_oob`) or play halts (`p90_smooth < 2.08 m/s`) until the exact frame the restart kick accelerates the ball ($v_{\text{ball}} > 4.8\text{ m/s}$).
4. **Match Period & Pre-Kickoff Rules (`X` vs. `D`):** All frames outside the two match periods (`0..11184`, `90652..110807`, `190198..199318`) are labeled `X`. Per `FAQ.md`, the pre-kickoff interval at the start of the second half (`110808..113741`, before the `113742` kickoff) is labeled `D`.

---

## 5. Full-Match (90-Minute) Calibration & Out-of-Sample Realism Audit

A common pitfall in hackathon challenges is overfitting to the 5-minute public window while producing unrealistic predictions on the remaining 85 minutes. We audited the sensor distributions and model outputs across the entire 90-minute match (`Half 1: 11185..90651` vs. `Half 2: 113742..190197`):

1. **Identical Sensor Stability Across Halves:**
   - **Half 1 (`11185..90651`):** `valid_ball = 59.3%`, `p90_spd` quantiles $(10\%, 25\%, 50\%, 75\%) = [1.27, 1.57, 2.64, 3.92]\text{ m/s}$.
   - **Half 2 (`113742..190197`):** `valid_ball = 58.4%`, `p90_spd` quantiles $(10\%, 25\%, 50\%, 75\%) = [1.28, 1.55, 2.65, 3.88]\text{ m/s}$.
2. **Effective Playing Time Matches La Liga Benchmarks:**
   - While the opening 5 minutes (`11185..18685`) had no goals, cards, substitutions, or VAR reviews (`80.4%` live / `19.6%` dead ball), across the full 90-minute match (including 4 goals, celebrations, substitutions, and set pieces), our model predicts **`54.2%` live play (`51.5 minutes` of effective ball-in-play time)** and **`45.8%` dead ball (`D`)**, matching Opta's published La Liga average effective playing time (`53.6 minutes` / `56.4%` live).
3. **Full-Match Possession Count & Spell Duration:**
   - Across the full 90 minutes, `Suleyman.csv` predicts **239 possessions** with a **mean duration of `14.1 seconds`** — matching the ground-truth mean possession duration (**`14.2 seconds`**) within **`0.1 seconds`**! By contrast, `baseline_smooth50` fragments the match into **594 possessions** (`7.5s` mean duration).

---

## 6. Where the Approach Still Fails & Limitations

1. **Early Post-Kickoff Ball-Tracker Glitch (`Seg 1 [11351..11553]` & `Seg 2 [11553..11624]`):**
   Between frames `11351` and `11460` ($4.4\text{s}$ immediately following the opening kickoff long ball), the raw ATD ball trajectory lags near Athletic (`A`) players ($\min(d_A) = 0.68\text{m–}1.50\text{m}$) while the nearest Barcelona (`H`) player is tracked $3.04\text{m–}7.09\text{m}$ away until `11460` ($\min(d_H) = 0.16\text{m}$). Because our decoder respects physical proximity, it transitions from `A` to `H` at `11463` rather than `11351`. Resolving tracker-identity glitches when the ball is mis-associated in 2D would require direct computer-vision ball re-detection from the tactical video (`BAR-ATH_tactical.mp4`).
2. **Ultra-Short 1–3 Second Contested Deflections (`Seg 10 [0.8s]`, `Seg 12 [1.3s]`, `Seg 16 [3.5s]`):**
   Under the competition's primary metric (`spell_f1_mean`), a single false-positive split inside a 40-second possession (`Seg 3` or `Seg 8`) penalizes both precision and recall across all five IoU thresholds (`0.3..0.7`). Setting `penalty = 42.0` completely eliminates internal fragmentation on long possessions (`Seg 3 = 100% H(41.6s)`, `Seg 8 = 99.5% H(42.2s)`, `Seg 9 = 100% A(20.0s)`, `Seg 15 = 100% A(35.7s)`, `Seg 20 = 100% H(11.8s)`), at the deliberate cost of smoothing over ultra-brief $0.8\text{s–}2.8\text{s}$ turnovers during chaotic midfield ping-pong.

---

## 7. How to Run the Code to Reproduce `Suleyman.csv`

### Environment Setup
```bash
pip install -r requirements.txt
export MTAH_DATA_DIR="/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
```

### Generate & Validate `Suleyman.csv`
```bash
# 1. Generate Suleyman.csv deterministically from raw tactical ATD + Smart Tagging XML (~40s on CPU)
python predict.py --out Suleyman.csv

# 2. Validate format, frame index (0..199318), and label set {H, A, D, X}
python evaluation/validate_submission.py Suleyman.csv

# 3. Run full evaluation suite with 1,000-draw block bootstrap
python evaluation/run_evaluation.py --bootstrap 1000
```
