def seat_risk(prediction: float, capacity: int, tight_utilization: float) -> dict:
    utilization = prediction / capacity
    if prediction >= capacity:
        risk = 'critical'
    elif utilization >= tight_utilization:
        risk = 'tight'
    else:
        risk = 'normal'
    return {
        'utilization': utilization,
        'headroom_seats': capacity - prediction,
        'shortage_seats': max(0.0, prediction - capacity),
        'risk': risk,
    }
