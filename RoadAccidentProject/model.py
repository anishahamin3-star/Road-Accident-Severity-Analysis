def predict_severity(weather, visibility, traffic_density, temperature):
    score = 0

    if weather == "fog":
        score += 2
    elif weather == "rain":
        score += 1

    if visibility == "low":
        score += 2
    elif visibility == "medium":
        score += 1

    if traffic_density == "high":
        score += 2
    elif traffic_density == "medium":
        score += 1

    if temperature >= 32:
        score += 1

    if score >= 5:
        return "fatal"
    elif score >= 3:
        return "major"
    else:
        return "minor"