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
    eff_scale_m = (eff_m ** gamma) if (eff_m > 0 and gamma != 1.0) else (eff_m if gamma == 1.0 else 1.0)
    eff_scale_live = (eff_live ** gamma) if (eff_live > 0 and gamma != 1.0) else (eff_live if gamma == 1.0 else 1.0)

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

    # Exit / Diminishing-Return Regime strictly in our trade's favor
    if eval_win_m >= LOGISTIC_HALF_HIGH:
        regime = "HALF-SLOPE"
    elif eval_win_m >= LOGISTIC_DECEL_HIGH:
        regime = "DECEL-EXIT"
    elif eval_win_m < win_prob_m - 1e-9:
        regime = "AGAINST"
    else:
        regime = "ACTIVE"

    dist_to_decel = LOGISTIC_DECEL_HIGH - eval_win_m
    dist_to_half = LOGISTIC_HALF_HIGH - eval_win_m
    target_win_m = min(win_prob_bal, LOGISTIC_DECEL_HIGH)
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


def calculate_base_ego(history, verbose=True):
    """Calculates historical Kelly-weighted Brier score calibration."""
    resolved = history.get("resolved", {})
    if not resolved:
        return 0.50
        
    bs_u_total, bs_m_total = 0.0, 0.0
    total_weight = 0.0
    
    for market_preds in resolved.values():
        market_preds = ensure_list(market_preds)
        for data in market_preds:
            outcome = data.get("outcome", 0)
            weight = data.get("kelly", 0.0)
            
            if weight <= 1e-9: 
                continue 
                
            bs_u_total += weight * ((data.get("pu", 0.5) - outcome) ** 2)
            bs_m_total += weight * ((data.get("pm", 0.5) - outcome) ** 2)
            total_weight += weight
            
    if total_weight == 0:
        return 0.50

    bs_u_total /= total_weight
    bs_m_total /= total_weight
        
    if verbose:
        print(f"[*] Current Brier Score - User: {bs_u_total:.4f} | Market: {bs_m_total:.4f}")
    
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
        if isinstance(apy, complex): apy = -1.0
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


def calculate_allocation(lower_bound, upper_bound, pm_bid, pm_ask, fee_rate, bankroll, base_ego, custom_min_edge=None):
    """Translates belief bounds into trade direction (YES/NO), true price, Kelly sizing, and edge."""
    if custom_min_edge is None:
        custom_min_edge = cfg.MIN_EDGE
    
    pu_mid = (lower_bound + upper_bound) / 2.0
    u_spread = upper_bound - lower_bound
    m_spread = pm_ask - pm_bid
    pm_mid = (pm_bid + pm_ask) / 2.0
    
    dynamic_ego = compute_dynamic_ego(base_ego, u_spread, m_spread)
    lm = calculate_logistic_metrics(pu_mid, pm_mid, dynamic_ego, fee_rate=fee_rate)
    pu_norm = lm["pu_norm"]
    
    cost_yes, cost_no = pm_ask, 1.0 - pm_bid 
    if pu_norm > cost_yes:
        action, base_price, win_prob = "YES", cost_yes, pu_norm
    elif (1.0 - pu_norm) > cost_no:
        action, base_price, win_prob = "NO", cost_no, 1.0 - pu_norm
    else:
        return "NONE", 0.0, 0.0, 0.0, 0.0, dynamic_ego, 0.0
    
    true_price = base_price + (fee_rate * base_price * (1 - base_price))
    edge = win_prob - true_price

    raw_kelly = max(0.0, edge / (1 - true_price)) if true_price < 1.0 else 0.0
    
    # Balanced / Efficiency Kelly: Scale raw Kelly by logistic marginal efficiency [4p(1-p)]^gamma
    # Defaults to gamma=0.5 (Square-Root Fisher information geodesic) for optimal growth-variance balance
    gamma = float(getattr(cfg, "KELLY_GAMMA", 0.5))
    eff_scale = (lm["eff_m"] ** gamma) if (lm["eff_m"] > 0 and gamma != 1.0) else (lm["eff_m"] if gamma == 1.0 else 1.0)
    eff_kelly = eff_scale * raw_kelly
    final_allocation = eff_kelly * bankroll
    
    if edge < custom_min_edge:
        return "THIN_EDGE", true_price, raw_kelly, eff_kelly, 0.0, dynamic_ego, edge
    
    return action, true_price, raw_kelly, eff_kelly, final_allocation, dynamic_ego, edge


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

