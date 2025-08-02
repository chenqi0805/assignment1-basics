import math


def learning_rate_schedule(
        t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int):
    if t < t_w:
        return t / t_w * alpha_max
    elif t >= t_w and t < t_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (alpha_max - alpha_min)
    else:
        return alpha_min