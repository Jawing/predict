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
    """Calculates conviction-weighted ego based on uncertainty spread."""
    u_conviction = base_ego * (1.0 - u_spread)
    m_conviction = (1.0 - base_ego) * (1.0 - m_spread)
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


def calculate_logistic_metrics(pu, pm, ego=0.5, held_kelly=None, cur_pm=None, fee_rate=0.0):
    """Computes Bayesian pooled probability p_norm, active edge, true cost, live Kelly, and logistic landmarks."""
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

    is_yes = pu >= pm
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

    base_k = max(0.0, float(held_kelly)) if held_kelly is not None else model_kelly
    if base_k < 1e-9:
        base_k = 0.0
    eff_kelly = eff_scale_live * base_k

    # Pure Unconstrained Bayesian Target (no artificial landmark clamping)
    target_win_m = win_prob_bal

    if eval_win_m >= target_win_m or live_active_edge <= 0.0:
        regime = "OVER-TGT"
    elif eval_win_m < win_prob_m - 1e-9:
        regime = "AGAINST"
    else:
        regime = "ACTIVE"

    dist_to_decel = LOGISTIC_DECEL_HIGH - eval_win_m
    dist_to_half = LOGISTIC_HALF_HIGH - eval_win_m
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
        "regime": regime,
        "dist_to_decel": dist_to_decel,
        "dist_to_half": dist_to_half
    }


def calculate_dynamic_kelly_action(win_prob_bal, cur_p, held_frac, bankroll, fee_rate=0.0, half_spread=0.005, gamma=None):
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

    half_sp = max(0.001, float(half_spread))

    # 1. Optimal Live Hold Target f*_hold (at current live price with true fees)
    hold_true_cost = cur_p + (fee_rate * cur_p * (1.0 - cur_p))
    edge_hold = p_norm - hold_true_cost
    if edge_hold > 0.0 and hold_true_cost < 1.0:
        raw_k_hold = edge_hold / (1.0 - hold_true_cost)
        eff_scale_hold = calculate_efficiency_scale(4.0 * cur_p * (1.0 - cur_p), gamma)
        f_star_hold = eff_scale_hold * raw_k_hold
    else:
        f_star_hold = 0.0

    # 2. Optimal Add Target f*_add (evaluated after crossing ask spread and paying fees)
    p_ask = min(0.999, cur_p + half_sp)
    buy_true_cost = p_ask + (fee_rate * p_ask * (1.0 - p_ask))
    edge_add = p_norm - buy_true_cost
    if edge_add > 0.0 and buy_true_cost < 1.0:
        raw_k_add = edge_add / (1.0 - buy_true_cost)
        eff_scale_add = calculate_efficiency_scale(4.0 * p_ask * (1.0 - p_ask), gamma)
        f_star_add = eff_scale_add * raw_k_add
    else:
        f_star_add = 0.0

    double_f_hold = 2.0 * f_star_hold

    # 3. Action Decision
    if dist_to_tgt <= 0.0 or edge_hold <= 0.0:
        # Target reached or negative edge at current price
        action_type = "SELL"
        signal_code = "SELL"
        action_label = f"SELL: OVER-TGT ({dist_to_tgt*100:+.1f}%)"
        delta_usd = held_frac * bankroll
    elif dist_to_tgt <= half_sp or f_star_hold < 0.005:
        # Exit Runway / Finish Line: Within orderbook half-spread of target, or remaining optimal holding < 0.5% bankroll
        # Remaining edge cannot cover spread friction; risk/reward is exhausted
        action_type = "TAKE_PROFIT"
        signal_code = "EXIT"
        action_label = f"TAKE PROFIT (+{dist_to_tgt*100:.1f}%)"
        delta_usd = held_frac * bankroll
    elif f_star_add > held_frac and (f_star_add - held_frac) >= 0.005:
        # Genuinely under-allocated: even at the Ask price (after paying spread), Kelly dictates adding >= 0.5% bankroll
        action_type = "ADD"
        signal_code = "ADD"
        add_frac = f_star_add - held_frac
        add_usd = add_frac * bankroll
        action_label = f"ADD +${add_usd:,.0f} (+{add_frac*100:.1f}%)"
        delta_usd = add_usd
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
        "double_f_hold": double_f_hold,
        "held_frac": held_frac,
        "delta_usd": delta_usd,
        "edge_hold": edge_hold,
        "edge_add": edge_add,
        "dist_to_tgt": dist_to_tgt
    }


def calculate_base_ego(history, verbose=True, weighting_mode=None):
    """
    Calculates historical Brier score calibration across resolved and exited positions.
    Weighting mode:
      - 'kelly' (default): Weights by theoretical model Kelly fraction.
      - 'capital': Weights by actual risked capital % (cost basis / bankroll).
      - 'equal': Equal weighting (1 vote per position).
    Supports skipping markets tagged with data.get("exclude_from_ego") == True.
    """
    if weighting_mode is None:
        weighting_mode = getattr(cfg, "EGO_WEIGHTING", "capital")

    resolved = history.get("resolved", {})
    exited = history.get("exited", {})
    if not resolved and not exited:
        return 0.50

    bs_u_total, bs_m_total = 0.0, 0.0
    total_weight = 0.0

    bankroll = cfg.BANKROLL if cfg.BANKROLL > 0 else 100.0

    def compute_weight(data):
        if data.get("exclude_from_ego"):
            return 0.0
        if weighting_mode == "equal":
            return 1.0
        elif weighting_mode == "kelly":
            wk = float(data.get("kelly", 0.0) or 0.0)
            if wk > 1e-9:
                return wk
            cb = float(data.get("cost_basis", 0.0) or 0.0)
            return max(0.01, cb / bankroll)
        else:  # "capital" (Actual Risked Capital % - True Kelly Allocation)
            if data.get("actual_kelly") is not None and float(data.get("actual_kelly")) > 0:
                return float(data.get("actual_kelly"))
            cb = float(data.get("cost_basis", 0.0) or 0.0)
            tot_s = float(data.get("total_size", 0.0) or 0.0)
            avg_p = float(data.get("avg_price", 0.0) or 0.0)
            pnl = float(data.get("realized_pnl", 0.0) or 0.0)
            act_dlrs = cb if cb > 0 else (tot_s * avg_p if tot_s > 0 else abs(pnl))
            return max(0.005, min(1.0, act_dlrs / bankroll))

    # 1. Terminal Resolutions (y in {0.0, 0.5, 1.0})
    for market_preds in resolved.values():
        for data in ensure_list(market_preds):
            weight = compute_weight(data)
            if weight <= 0.0:
                continue
            outcome = float(data.get("outcome", 0.0) or 0.0)
            pu = float(data.get("pu", 0.5))
            pm = float(data.get("pm", 0.5))
            bs_u_total += weight * ((pu - outcome) ** 2)
            bs_m_total += weight * ((pm - outcome) ** 2)
            total_weight += weight

    # 2. Exited Positions (Synthetic Repricing Resolution against exit price)
    for market_preds in exited.values():
        for data in ensure_list(market_preds):
            weight = compute_weight(data)
            if weight <= 0.0:
                continue
            exit_price = data.get("exit_price") if data.get("exit_price") is not None else data.get("cur_pm")
            if exit_price is None:
                continue
            exit_outcome = float(exit_price)
            pu_raw = data.get("pu")
            pm = float(data.get("pm", 0.5))
            r_pnl = float(data.get("realized_pnl", 0.0) or 0.0)

            is_uncalibrated_pre_program = (pu_raw is None or abs(float(pu_raw) - pm) < 1e-6)

            if is_uncalibrated_pre_program and r_pnl < 0:
                bs_u_total += weight * ((pm - exit_outcome) ** 2)
                bs_m_total += weight * 0.0
            else:
                pu = float(pu_raw if pu_raw is not None else pm)
                bs_u_total += weight * ((pu - exit_outcome) ** 2)
                bs_m_total += weight * ((pm - exit_outcome) ** 2)
            total_weight += weight

    if total_weight == 0:
        return 0.50

    bs_u_total /= total_weight
    bs_m_total /= total_weight

    if verbose:
        mode_tag = f"[{weighting_mode.upper()}]"
        print(f"[*] Base Ego {mode_tag} - User Brier: {bs_u_total:.4f} | Market Brier: {bs_m_total:.4f}")

    if bs_u_total == 0 and bs_m_total == 0:
        return 0.50
    return bs_m_total / (bs_u_total + bs_m_total)


def calculate_trade_ego_metrics(data, cat="resolved", bankroll=100.0, weighting_mode=None):
    """Calculates individual trade Brier contributions: weight, bs_u, bs_m, ego_drag, and net_adv."""
    if weighting_mode is None:
        weighting_mode = getattr(cfg, "EGO_WEIGHTING", "kelly")
    
    bankroll = bankroll if bankroll > 0 else 100.0
    
    # Weight calculation
    if data.get("exclude_from_ego"):
        weight = 0.0
    elif weighting_mode == "equal":
        weight = 1.0
    elif weighting_mode == "kelly":
        wk = float(data.get("kelly", 0.0) or 0.0)
        if wk > 1e-9:
            weight = wk
        else:
            cb = float(data.get("cost_basis", 0.0) or 0.0)
            weight = max(0.01, cb / bankroll)
    else:  # capital
        if data.get("actual_kelly") is not None and float(data.get("actual_kelly")) > 0:
            weight = float(data.get("actual_kelly"))
        else:
            cb = float(data.get("cost_basis", 0.0) or 0.0)
            tot_s = float(data.get("total_size", 0.0) or 0.0)
            avg_p = float(data.get("avg_price", 0.0) or 0.0)
            pnl = float(data.get("realized_pnl") if data.get("realized_pnl") is not None else data.get("pnl", 0.0) or 0.0)
            act_dlrs = cb if cb > 0 else (tot_s * avg_p if tot_s > 0 else abs(pnl))
            weight = max(0.005, min(1.0, act_dlrs / bankroll))
            
    pm = float(data.get("pm", 0.5) or 0.5)
    pu_raw = data.get("pu")
    
    if cat == "resolved":
        outcome = float(data.get("outcome", 0.0) or 0.0)
        pu = float(pu_raw if pu_raw is not None else 0.5)
        bs_u = (pu - outcome) ** 2
        bs_m = (pm - outcome) ** 2
    else:  # exited
        exit_price = data.get("exit_price") if data.get("exit_price") is not None else data.get("cur_pm")
        exit_outcome = float(exit_price if exit_price is not None else pm)
        is_uncal = (pu_raw is None or abs(float(pu_raw) - pm) < 1e-6)
        r_pnl = float(data.get("realized_pnl") if data.get("realized_pnl") is not None else data.get("pnl", 0.0) or 0.0)
        if is_uncal and r_pnl < 0:
            bs_u = (pm - exit_outcome) ** 2
            bs_m = 0.0
        else:
            pu = float(pu_raw if pu_raw is not None else pm)
            bs_u = (pu - exit_outcome) ** 2
            bs_m = (pm - exit_outcome) ** 2

    ego_drag = weight * (bs_u - bs_m)
    net_adv = -ego_drag
    return {
        "weight": weight,
        "bs_u": bs_u,
        "bs_m": bs_m,
        "ego_drag": ego_drag,
        "net_adv": net_adv,
    }


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
    m_spread = pm_ask - pm_bid
    pm_mid = (pm_bid + pm_ask) / 2.0
    
    dynamic_ego = compute_dynamic_ego(base_ego, u_spread, m_spread)
    lm = calculate_logistic_metrics(pu_mid, pm_mid, dynamic_ego, held_kelly=held_kelly, fee_rate=fee_rate)
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
            price = float(ask['price'])
            size = float(ask['size'])
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
