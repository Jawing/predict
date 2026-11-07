"""Mathematical foundation: Bayesian logit pooling, dynamic ego, logistic efficiency, and Kelly criterion."""
import math
import re
import numpy as np
import config as cfg
from utils import ensure_list

# --- LOGISTIC DIMINISHING-RETURN LANDMARKS ---
# For sigma(z) = 1 / (1 + e^-z), normalized slope efficiency is eta(p) = 4*p*(1 - p) in [0, 1]
LOGISTIC_DECEL_LOW = (3.0 - math.sqrt(3.0)) / 6.0   # ~21.13%: Max Rate of Deceleration (sigma'''(z)=0, eta=66.7%)
LOGISTIC_DECEL_HIGH = (3.0 + math.sqrt(3.0)) / 6.0  # ~78.87%: Max Rate of Deceleration (sigma'''(z)=0, eta=66.7%)
LOGISTIC_HALF_LOW = (2.0 - math.sqrt(2.0)) / 4.0    # ~14.64%: Half-Maximum Slope (sigma'(z)=0.5*sigma'(0), eta=50.0%)
LOGISTIC_HALF_HIGH = (2.0 + math.sqrt(2.0)) / 4.0   # ~85.36%: Half-Maximum Slope (sigma'(z)=0.5*sigma'(0), eta=50.0%)


def compute_dynamic_ego(base_ego, u_spread, m_spread=0.0):
    """Calculates conviction-weighted ego based on user uncertainty spread."""
    u_spread = max(0.0, min(1.0, float(u_spread or 0.0)))
    base_ego = max(0.01, min(0.99, float(base_ego or 0.5)))
    u_conviction = base_ego * (1.0 - u_spread)
    m_conviction = 1.0 - base_ego
    if (u_conviction + m_conviction) > 0:
        return u_conviction / (u_conviction + m_conviction)
    return base_ego


def calculate_efficiency_scale(eff, gamma=None):
    """Calculates logistic efficiency scaling factor [4p(1-p)]^gamma."""
    if gamma is None:
        gamma = float(getattr(cfg, "KELLY_GAMMA", 0.5))
    if gamma == 0.0:
        return 1.0
    if eff <= 0.0:
        return 0.0
    if gamma == 1.0:
        return eff
    return eff ** gamma


def calculate_logistic_metrics(pu, pm, ego=0.5, held_kelly=None, cur_pm=None, fee_rate=0.0, raw_yes_space=False):
    """Computes Bayesian pooled probability p_norm, active edge, true cost, live Kelly, and logistic landmarks.

    When raw_yes_space=False (default), pu, pm, and cur_pm are already oriented in the held/evaluated
    position's coordinate frame. When raw_yes_space=True (used during initial side selection in
    calculate_allocation), is_yes = (pu >= pm) selects whether YES or NO has positive edge.
    """
    eps = 0.01
    pu = float(pu if pu is not None else 0.5)
    pm = float(pm if pm is not None else 0.5)
    ego = float(ego if ego is not None else 0.5)
    eval_pm = float(cur_pm if cur_pm is not None else pm)

    if abs(pu - pm) < 1e-9:
        pu_norm = pm
    else:
        pu_clip = float(np.clip(pu, eps, 1.0 - eps))
        pm_clip = float(np.clip(pm, eps, 1.0 - eps))
        logit_pu = math.log(pu_clip / (1.0 - pu_clip))
        logit_pm = math.log(pm_clip / (1.0 - pm_clip))
        logit_norm = (ego * logit_pu) + ((1.0 - ego) * logit_pm)
        pu_norm = 1.0 / (1.0 + math.exp(-logit_norm))

    is_yes = (pu >= pm) if raw_yes_space else True
    win_prob_bal = pu_norm if is_yes else (1.0 - pu_norm)
    win_prob_u = pu if is_yes else (1.0 - pu)
    win_prob_m = pm if is_yes else (1.0 - pm)
    eval_win_m = eval_pm if is_yes else (1.0 - eval_pm)

    # Logistic slope efficiency eta(p) = 4*p*(1-p)
    eff_m = 4.0 * max(0.0, min(1.0, pm)) * (1.0 - max(0.0, min(1.0, pm)))
    eff_live = 4.0 * max(0.0, min(1.0, eval_pm)) * (1.0 - max(0.0, min(1.0, eval_pm)))
    eff_norm = 4.0 * pu_norm * (1.0 - pu_norm)

    gamma = float(getattr(cfg, "KELLY_GAMMA", 0.5))
    eff_scale_m = calculate_efficiency_scale(eff_m, gamma)
    eff_scale_live = calculate_efficiency_scale(eff_live, gamma)

    true_cost = win_prob_m + (fee_rate * win_prob_m * (1.0 - win_prob_m))
    active_edge = win_prob_bal - true_cost
    if abs(active_edge) < 1e-9:
        active_edge = 0.0
    model_kelly = max(0.0, active_edge / (1.0 - true_cost)) if (true_cost < 1.0 and active_edge > 0.0) else 0.0
    eff_model_kelly = eff_scale_m * model_kelly

    live_true_cost = eval_win_m + (fee_rate * eval_win_m * (1.0 - eval_win_m))
    live_active_edge = win_prob_bal - live_true_cost
    if abs(live_active_edge) < 1e-9:
        live_active_edge = 0.0
    live_model_kelly = max(0.0, live_active_edge / (1.0 - live_true_cost)) if (live_true_cost < 1.0 and live_active_edge > 0.0) else 0.0
    eff_live_model_kelly = eff_scale_live * live_model_kelly

    if held_kelly is not None:
        # held_kelly is already an effective Kelly fraction; do not multiply by eff_scale a second time
        eff_kelly = max(0.0, float(held_kelly))
        if eff_kelly < 1e-9:
            eff_kelly = 0.0
        base_k = (eff_kelly / eff_scale_m) if eff_scale_m > 1e-9 else eff_kelly
    else:
        base_k = model_kelly
        eff_kelly = eff_live_model_kelly

    # Pure Unconstrained Bayesian Target (no artificial landmark clamping)
    target_win_m = win_prob_bal

    if eval_win_m >= target_win_m or live_active_edge <= 0.0:
        regime = "OVER-TGT"
    elif eval_win_m < win_prob_m - 1e-9:
        regime = "AGAINST"
    else:
        regime = "ACTIVE"

    dist_to_target = target_win_m - eval_win_m
    delta_kelly = live_model_kelly - base_k
    if abs(delta_kelly) < 1e-9:
        delta_kelly = 0.0

    return {
        "pu_norm": pu_norm,
        "is_yes": is_yes,
        "win_prob_bal": win_prob_bal,
        "win_prob_u": win_prob_u,
        "win_prob_m": win_prob_m,
        "eval_pm": eval_pm,
        "eval_win_m": eval_win_m,
        "target_win_m": target_win_m,
        "dist_to_target": dist_to_target,
        "delta_kelly": delta_kelly,
        "eff_m": eff_m,
        "eff_live": eff_live,
        "eff_norm": eff_norm,
        "active_edge": active_edge,
        "live_active_edge": live_active_edge,
        "model_kelly": model_kelly,
        "eff_model_kelly": eff_model_kelly,
        "live_model_kelly": live_model_kelly,
        "eff_live_model_kelly": eff_live_model_kelly,
        "eff_kelly": eff_kelly,
        "eff_scale_m": eff_scale_m,
        "eff_scale_live": eff_scale_live,
        "gamma": gamma,
        "regime": regime
    }


def calculate_dynamic_kelly_action(win_prob_bal, cur_p, held_frac, bankroll, fee_rate=0.0, half_spread=0.005, gamma=None, entry_kelly=None, best_bid=None, best_ask=None):
    """
    Evaluates position holding against dynamic Kelly optimization,
    accounting for downside risk, crossing the bid-ask spread, and
    discrete conviction-based holding without premature continuous trimming.
    """
    if gamma is None:
        gamma = float(getattr(cfg, "KELLY_GAMMA", 0.5))

    p_norm = float(win_prob_bal)
    cur_p = max(0.001, min(0.999, float(cur_p)))
    held_frac = max(0.0, float(held_frac))
    dist_to_tgt = p_norm - cur_p

    half_sp = max(0.0005, float(half_spread))
    # f_star_add already prices in crossing the Ask spread and fees;
    # add_hurdle is a pure 0.5% bankroll dust filter so Bid noise never suppresses valid Ask conviction.
    add_hurdle = 0.005

    # 1. Optimal Live Hold Target f*_hold (for shares already owned; no re-entry fee to hold to resolution)
    edge_hold = p_norm - cur_p
    if edge_hold > 0.0 and cur_p < 1.0:
        raw_k_hold = edge_hold / (1.0 - cur_p)
        eff_scale_hold = calculate_efficiency_scale(4.0 * cur_p * (1.0 - cur_p), gamma)
        f_star_hold = eff_scale_hold * raw_k_hold
    else:
        f_star_hold = 0.0

    # 2. Optimal Add Target f*_add (evaluated after crossing ask spread and paying fees)
    p_ask = min(0.999, float(best_ask) if (best_ask is not None and float(best_ask) > 0) else (cur_p + half_sp))
    buy_true_cost = p_ask + (fee_rate * p_ask * (1.0 - p_ask))
    edge_add = p_norm - buy_true_cost
    if edge_add > 0.0 and buy_true_cost < 1.0:
        raw_k_add = edge_add / (1.0 - buy_true_cost)
        eff_scale_add = calculate_efficiency_scale(4.0 * p_ask * (1.0 - p_ask), gamma)
        f_star_add = eff_scale_add * raw_k_add
    else:
        f_star_add = 0.0

    # Anchored holding reference: respects the entry/thesis Kelly allocation
    entry_ref = max(f_star_hold, float(entry_kelly or 0.0))
    double_f_hold = 2.0 * entry_ref
    add_frac = f_star_add - held_frac
    # Effective actionable Kelly across the two-boundary no-trade zone:
    # - Below f*_add: target is f*_add (scales up to Ask+fee boundary)
    # - Inside [f*_add, f*_hold]: target is held_frac (sunk entry fee/spread holds current size without phantom exposure)
    # - Above f*_hold: target is f*_hold (decayed live hold conviction as price approaches target)
    if f_star_add <= f_star_hold:
        f_star_eff = min(f_star_hold, max(f_star_add, held_frac))
    else:
        f_star_eff = f_star_add

    # 3. Action Decision
    p_bid = max(0.001, float(best_bid) if (best_bid is not None and float(best_bid) > 0) else (cur_p - half_sp))
    if dist_to_tgt <= 0.0:
        # Target reached or exceeded at current price
        action_type = "SELL"
        signal_code = "SELL"
        action_label = f"SELL: OVER-TGT ({dist_to_tgt*100:+.1f}%)"
        delta_usd = held_frac * bankroll
    elif (p_norm - p_bid) <= 0.005 or f_star_hold < 0.005:
        # Exit Runway / Finish Line: Executable Bid is within 0.5c of target, or remaining optimal holding < 0.5% bankroll
        action_type = "TAKE_PROFIT"
        signal_code = "EXIT"
        action_label = f"TAKE PROFIT (+{dist_to_tgt*100:.1f}%)"
        delta_usd = held_frac * bankroll
    elif add_frac >= add_hurdle:
        # Genuinely under-allocated: even at the Ask price (after paying spread), Kelly dictates adding >= hurdle
        action_type = "ADD"
        signal_code = "ADD"
        add_usd = add_frac * bankroll
        action_label = f"ADD +${add_usd:,.0f} (+{add_frac*100:.1f}%)"
        delta_usd = add_usd
    elif held_frac > double_f_hold:
        # Position has grown beyond twice the authorized entry/holding conviction: trim excess back down
        trim_frac = held_frac - entry_ref
        trim_usd = trim_frac * bankroll
        action_type = "TRIM"
        signal_code = "TRIM"
        action_label = f"TRIM -${trim_usd:,.0f} (-{trim_frac*100:.1f}%)"
        delta_usd = trim_usd
    else:
        # Conviction Holding: let winning positions run to target without premature spread-dragging micro-trims
        action_type = "HOLD"
        signal_code = "HOLD"
        action_label = f"HOLD (+{dist_to_tgt*100:.1f}% to exit)"
        delta_usd = 0.0

    return {
        "action_type": action_type,
        "signal_code": signal_code,
        "action_label": action_label,
        "f_star_hold": f_star_hold,
        "f_star_add": f_star_add,
        "f_star_eff": f_star_eff,
        "double_f_hold": double_f_hold,
        "held_frac": held_frac,
        "delta_usd": delta_usd,
        "edge_hold": edge_hold,
        "edge_add": edge_add,
        "dist_to_tgt": dist_to_tgt
    }


def compute_trajectory_brier(data, outcome):
    """
    Computes Brier scores across multi-snapshot update history weighted by each snapshot's Kelly conviction,
    deduplicating consecutive snapshots where user belief pu did not change.
    If no snapshots or single snapshot, evaluates direct contemporaneous values.
    """
    outcome = float(outcome)
    snapshots = data.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) <= 1:
        pu = float(data.get("pu", 0.5) if data.get("pu") is not None else 0.5)
        pm = float(data.get("pm", 0.5) if data.get("pm") is not None else 0.5)
        return (pu - outcome) ** 2, (pm - outcome) ** 2

    valid_snaps = []
    for s in snapshots:
        if isinstance(s, dict) and "pu" in s and "pm" in s:
            try:
                pu_s = float(s["pu"])
                pm_s = float(s["pm"])
                k_s = max(0.001, float(s.get("kelly") or s.get("entry_kelly") or 0.001))
                if valid_snaps and abs(valid_snaps[-1][0] - pu_s) < 1e-6:
                    continue
                valid_snaps.append((pu_s, pm_s, k_s))
            except (ValueError, TypeError):
                continue

    if not valid_snaps:
        pu = float(data.get("pu", 0.5) if data.get("pu") is not None else 0.5)
        pm = float(data.get("pm", 0.5) if data.get("pm") is not None else 0.5)
        return (pu - outcome) ** 2, (pm - outcome) ** 2

    tot_k = sum(k_s for _, _, k_s in valid_snaps)
    bs_u = sum(k_s * ((pu_s - outcome) ** 2) for pu_s, _, k_s in valid_snaps) / tot_k
    bs_m = sum(k_s * ((pm_s - outcome) ** 2) for _, pm_s, k_s in valid_snaps) / tot_k
    return bs_u, bs_m


def calculate_trade_weight(data, bankroll=100.0, is_live_wallet=False):
    """Calculates trade calibration weight bounded by min(k_model, k_executed)."""
    if data.get("exclude_from_ego") or data.get("is_watch") or data.get("action") == "WATCH":
        return 0.0
    if data.get("actual_kelly") is not None and float(data.get("actual_kelly")) > 0:
        return float(data.get("actual_kelly"))
    wk = float(data.get("entry_kelly") or data.get("kelly", 0.0) or 0.0)
    eb = float(data.get("entry_bankroll") or bankroll or 100.0)
    if eb <= 0:
        eb = 100.0
    ec = float(data.get("entry_cost") or data.get("cost_basis") or 0.0)
    if ec <= 0:
        tot_s = float(data.get("total_size", 0.0) or float(data.get("shares", 0.0) or 0.0))
        avg_p = float(data.get("avg_price", 0.0) or float(data.get("entry_price", 0.0) or 0.0))
        ec = tot_s * avg_p
    k_exec = (ec / eb) if ec > 0 else 0.0
    if is_live_wallet and data.get("source") == "local" and k_exec <= 1e-9:
        return 0.0
    if wk > 1e-9 and k_exec > 1e-9:
        return min(wk, k_exec)
    if wk > 1e-9:
        return wk
    if k_exec > 1e-9:
        return min(1.0, k_exec)
    pnl = float(data.get("realized_pnl") if data.get("realized_pnl") is not None else data.get("pnl", 0.0) or 0.0)
    return max(0.001, min(1.0, abs(pnl) / eb))


def calculate_trade_ego_metrics(data, cat="resolved", bankroll=100.0, is_live_wallet=False):
    """Calculates individual trade Brier contributions: weight, bs_u, bs_m, ego_drag, and net_adv."""
    bankroll = bankroll if bankroll > 0 else 100.0
    weight = calculate_trade_weight(data, bankroll=bankroll, is_live_wallet=is_live_wallet)
            
    pm = float(data.get("pm", 0.5) or 0.5)
    pu_raw = data.get("pu")
    
    if cat == "resolved":
        outcome = float(data.get("outcome", 0.0) or 0.0)
        bs_u, bs_m = compute_trajectory_brier(data, outcome)
    else:  # exited
        exit_price = data.get("exit_price") if data.get("exit_price") is not None else data.get("cur_pm")
        exit_outcome = float(exit_price if exit_price is not None else pm)
        is_uncal = (pu_raw is None or abs(float(pu_raw) - pm) < 1e-6)
        r_pnl = float(data.get("realized_pnl") if data.get("realized_pnl") is not None else data.get("pnl", 0.0) or 0.0)
        if is_uncal and r_pnl < 0:
            bs_u = (pm - exit_outcome) ** 2
            bs_m = 0.0
        else:
            bs_u, bs_m = compute_trajectory_brier(data, exit_outcome)

    ego_drag = weight * (bs_u - bs_m)
    net_adv = -ego_drag
    return {
        "weight": weight,
        "bs_u": bs_u,
        "bs_m": bs_m,
        "ego_drag": ego_drag,
        "net_adv": net_adv,
    }


def calculate_base_ego(history, verbose=True):
    """
    Calculates historical Brier score calibration across resolved and exited positions,
    weighted by authorized Kelly capital allocation.
    Supports skipping markets tagged with data.get("exclude_from_ego") == True.
    """
    resolved = history.get("resolved", {})
    exited = history.get("exited", {})
    if not resolved and not exited:
        return 0.50

    bs_u_total, bs_m_total = 0.0, 0.0
    total_weight = 0.0
    ws = history.get("wallet_summary", {}) if isinstance(history, dict) else {}
    bankroll = float(ws.get("bankroll") or cfg.BANKROLL) if isinstance(ws, dict) else cfg.BANKROLL
    if bankroll <= 0:
        bankroll = 100.0
    is_live_w = bool(isinstance(history, dict) and str(history.get("profile_id") or history.get("wallet_address") or "").startswith("0x"))

    # 1. Terminal Resolutions (y in {0.0, 0.5, 1.0})
    for market_preds in resolved.values():
        for data in ensure_list(market_preds):
            m = calculate_trade_ego_metrics(data, cat="resolved", bankroll=bankroll, is_live_wallet=is_live_w)
            if m["weight"] <= 0.0:
                continue
            bs_u_total += m["weight"] * m["bs_u"]
            bs_m_total += m["weight"] * m["bs_m"]
            total_weight += m["weight"]

    # 2. Exited Positions (Synthetic Repricing Resolution against exit price)
    for market_preds in exited.values():
        for data in ensure_list(market_preds):
            exit_price = data.get("exit_price") if data.get("exit_price") is not None else data.get("cur_pm")
            if exit_price is None:
                continue
            m = calculate_trade_ego_metrics(data, cat="exited", bankroll=bankroll, is_live_wallet=is_live_w)
            if m["weight"] <= 0.0:
                continue
            bs_u_total += m["weight"] * m["bs_u"]
            bs_m_total += m["weight"] * m["bs_m"]
            total_weight += m["weight"]

    if total_weight == 0:
        return 0.50

    bs_u_total /= total_weight
    bs_m_total /= total_weight

    if verbose:
        print(f"[*] Base Ego [KELLY] - User Brier: {bs_u_total:.4f} | Market Brier: {bs_m_total:.4f}")

    if bs_u_total == 0 and bs_m_total == 0:
        return 0.50
    return bs_m_total / (bs_u_total + bs_m_total)


def calculate_annualized_yield(edge, true_price, days_until):
    """Calculates raw expected ROI and geometric compounding APY."""
    if days_until <= 0.0 or true_price <= 0 or abs(edge) < 1e-9:
        return 0.0, 0.0
        
    roi = edge / true_price
    try:
        apy = -1.0 if roi <= -1.0 else ((1 + roi) ** (365.0 / days_until)) - 1
        if isinstance(apy, complex):
            apy = -1.0
    except Exception:
        apy = 0.0
        
    return roi, apy


def parse_user_input(user_input):
    """Parses single percentage or range string (e.g., '42' or '36-48')."""
    cleaned = user_input.replace(' - ', ' ')
    cleaned = re.sub(r'(?<=\d)-', ' ', cleaned)
    cleaned = re.sub(r'(?<=%)-', ' ', cleaned)
    
    str_numbers = re.findall(r'-?\d+\.?\d*', cleaned)
    if not str_numbers:
        raise ValueError("Invalid input")
        
    numbers = []
    for s in str_numbers[:2]:
        if s.startswith('-'):
            raise ValueError(f"Negative values ({s}) are not allowed. Please enter 0-100.")
        num = float(s)
        if num < 0 or num > 100:
            raise ValueError(f"Value '{num}' is out of bounds. Percentages must be between 0 and 100")
        numbers.append(num)
        
    if len(numbers) == 1:
        return round(numbers[0] / 100.0, 6), round(numbers[0] / 100.0, 6)
    else:
        return round(min(numbers[0], numbers[1]) / 100.0, 6), round(max(numbers[0], numbers[1]) / 100.0, 6)


def calculate_allocation(lower_bound, upper_bound, pm_bid, pm_ask, fee_rate, bankroll, base_ego, custom_min_edge=None, held_kelly=None):
    """Translates belief bounds into trade direction (YES/NO), true price, Kelly sizing, edge, and logistic metrics."""
    if custom_min_edge is None:
        custom_min_edge = cfg.MIN_EDGE
    
    pu_mid = (lower_bound + upper_bound) / 2.0
    u_spread = upper_bound - lower_bound
    pm_mid = (pm_bid + pm_ask) / 2.0
    
    dynamic_ego = compute_dynamic_ego(base_ego, u_spread)
    lm = calculate_logistic_metrics(pu_mid, pm_mid, dynamic_ego, fee_rate=fee_rate, raw_yes_space=True)
    pu_norm = lm["pu_norm"]
    
    cost_yes, cost_no = pm_ask, 1.0 - pm_bid 
    if pu_norm > cost_yes:
        action, base_price, win_prob = "YES", cost_yes, pu_norm
    elif (1.0 - pu_norm) > cost_no:
        action, base_price, win_prob = "NO", cost_no, 1.0 - pu_norm
    else:
        return "NONE", 0.0, 0.0, 0.0, 0.0, dynamic_ego, 0.0, lm
    
    true_price = base_price + (fee_rate * base_price * (1 - base_price))
    edge = win_prob - true_price

    raw_kelly = max(0.0, edge / (1 - true_price)) if true_price < 1.0 else 0.0
    
    eff_scale = calculate_efficiency_scale(lm["eff_m"])
    eff_kelly = eff_scale * raw_kelly
    final_allocation = eff_kelly * bankroll
    
    if edge < custom_min_edge:
        return "THIN_EDGE", true_price, raw_kelly, eff_kelly, 0.0, dynamic_ego, edge, lm
    
    return action, true_price, raw_kelly, eff_kelly, final_allocation, dynamic_ego, edge, lm


def calculate_vwap(asks, target_usdc):
    """Walks the order book asks ascendingly to determine volume-weighted average price."""
    if not asks or target_usdc <= 0:
        return 0.0, 0.0
    valid_asks = []
    for ask in asks:
        try:
            if isinstance(ask, (list, tuple)):
                price, size = float(ask[0]), float(ask[1])
            elif isinstance(ask, dict):
                price, size = float(ask.get('price', 0)), float(ask.get('size', 0))
            else:
                continue
            if price > 0 and size > 0:
                valid_asks.append((price, size))
        except Exception:
            continue
    if not valid_asks:
        return 0.0, 0.0

    valid_asks.sort(key=lambda x: x[0])
    total_shares = 0.0
    remaining_usdc = target_usdc
    total_spent = 0.0
    for price, size in valid_asks:
        max_usdc_at_this_level = price * size
        if remaining_usdc >= max_usdc_at_this_level:
            total_shares += size
            remaining_usdc -= max_usdc_at_this_level
            total_spent += max_usdc_at_this_level
        else:
            shares = remaining_usdc / price
            total_shares += shares
            total_spent += remaining_usdc
            remaining_usdc = 0.0
            break
    if total_shares > 0:
        vwap = total_spent / total_shares
        return vwap, total_spent
    return 0.0, 0.0


def calculate_optimal_market_buy(asks, target_usdc, p_norm, fee_rate=0.0, min_edge=0.01):
    """
    Determines the optimal dollar amount to immediately market-buy without limit orders.
    Consumes orderbook asks ascendingly up to target_usdc, cutting off any asks where
    the marginal price exceeds the user's belief hurdle (p_ask + fee > p_norm - min_edge).
    Returns: (fill_usd, mean_vwap, vwap_edge)
    """
    if not asks or target_usdc <= 0:
        return 0.0, 0.0, 0.0
    valid_asks = []
    for ask in asks:
        try:
            if isinstance(ask, (list, tuple)):
                price, size = float(ask[0]), float(ask[1])
            elif isinstance(ask, dict):
                price, size = float(ask.get('price', 0)), float(ask.get('size', 0))
            else:
                continue
            if price > 0 and size > 0:
                valid_asks.append((price, size))
        except Exception:
            continue
    if not valid_asks:
        return 0.0, 0.0, 0.0

    valid_asks.sort(key=lambda x: x[0])
    cum_usd = 0.0
    cum_sh = 0.0

    for p, s in valid_asks:
        p_fee = p + (fee_rate * p * (1.0 - p))
        marg_edge = p_norm - p_fee
        if marg_edge < min_edge:
            break

        level_cost = p * s
        needed = target_usdc - cum_usd
        if needed <= 0:
            break

        take_cost = min(level_cost, needed)
        take_sh = take_cost / p
        cum_usd += take_cost
        cum_sh += take_sh
        if cum_usd >= target_usdc:
            break

    if cum_sh > 0:
        vwap = cum_usd / cum_sh
        vwap_fee = vwap + (fee_rate * vwap * (1.0 - vwap))
        vwap_edge = p_norm - vwap_fee
    else:
        cum_usd = 0.0
        vwap = valid_asks[0][0] if valid_asks else 0.0
        vwap_fee = vwap + (fee_rate * vwap * (1.0 - vwap))
        vwap_edge = p_norm - vwap_fee

    return round(cum_usd, 2), round(vwap, 4), round(vwap_edge, 4)


def restore_snapshot_to_prediction(p_target, prev_snap, base_ego=0.5):
    """Restores a prediction dict in-place to the state captured in prev_snap."""
    p_target["pu"] = float(prev_snap["pu"])
    p_target["pm"] = float(prev_snap["pm"])
    p_target["entry_price"] = float(prev_snap.get("entry_price") or prev_snap["pm"])
    if prev_snap.get("dynamic_ego") is not None:
        p_target["dynamic_ego"] = float(prev_snap["dynamic_ego"])
    if prev_snap.get("date") is not None:
        p_target["date"] = prev_snap["date"]
    if prev_snap.get("action") is not None:
        p_target["action"] = prev_snap["action"]
        p_target["is_watch"] = (prev_snap["action"] == "WATCH")

    if prev_snap.get("kelly") is not None:
        p_target["kelly"] = float(prev_snap["kelly"])
        p_target["entry_kelly"] = float(prev_snap.get("entry_kelly", prev_snap["kelly"]))
    elif prev_snap.get("entry_kelly") is not None:
        p_target["entry_kelly"] = float(prev_snap["entry_kelly"])
        p_target["kelly"] = float(prev_snap["entry_kelly"])
    if prev_snap.get("raw_kelly") is not None:
        p_target["raw_kelly"] = float(prev_snap["raw_kelly"])
    if prev_snap.get("edge") is not None:
        p_target["edge"] = float(prev_snap["edge"])
    if prev_snap.get("apy") is not None:
        p_target["apy"] = float(prev_snap["apy"])

    if prev_snap.get("kelly") is None or prev_snap.get("edge") is None:
        f_r = float(p_target.get("fee_rate", 0.0) or 0.0)
        ego_v = float(p_target.get("dynamic_ego", base_ego))
        lm_recalc = calculate_logistic_metrics(
            p_target["pu"], p_target["pm"], ego_v,
            cur_pm=p_target.get("cur_pm"), fee_rate=f_r
        )
        if prev_snap.get("kelly") is None:
            p_target["kelly"] = lm_recalc["eff_model_kelly"]
            if not p_target.get("entry_kelly"):
                p_target["entry_kelly"] = lm_recalc["eff_model_kelly"]
        if prev_snap.get("raw_kelly") is None:
            p_target["raw_kelly"] = lm_recalc["model_kelly"]
        if prev_snap.get("edge") is None:
            p_target["edge"] = lm_recalc["active_edge"]

