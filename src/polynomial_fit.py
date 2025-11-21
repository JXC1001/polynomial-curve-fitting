import math
import random

## Random k-value initialization
ks = {
    "k_0": random.uniform(-10, 10),
    "k_1": random.uniform(-10, 10),
    "k_2": random.uniform(-10, 10),
    "k_3": random.uniform(-10, 10),
    "k_4": random.uniform(-10, 10),
    "k_5": random.uniform(-10, 10)
}

## Save keys and original values for comparison later
k = list(ks.keys())
originals = list(ks.values())

## Create list that saves y-values
ys = []

## Get y-values from user
for i in range(7):
    x = i - 3
    ys.append(float(input("Y-value at " + str(x) + "?: ")))

def exec_function(r, k_values):
    keys = list(k_values.keys())
    predicted = []
    iterations = int(r * 2 + 1)
    for sample_index in range(iterations):
        y = 0
        x = sample_index - r
        x_norm = x / r
        for power_index in range(len(keys)):
            y += k_values[keys[power_index]] * math.pow(x_norm, power_index)
        predicted.append(y)
    return predicted

def calc_loss(predicted, y_values):
    total_loss = 0
    for i in range(len(y_values)):
        difference = y_values[i] - predicted[i]
        total_loss += math.pow(difference, 2)
    return total_loss

def dl_dki(k, y_values, predicted, r):
    ## Formula for partial derivatives: -2∑( a(x) - p(x) ) * x^j
    partial_derivative = 0
    for i in range(len(y_values)):
        x = i - r
        x_norm = x / r
        difference = y_values[i] - predicted[i]
        partial_derivative += -2 * difference * math.pow(x_norm, k)
    return partial_derivative

## Model running process as a function
def run(lr, k_values, y_values, iterations):       
    domain_radius = (len(y_values) - 1) / 2
    for i in range(iterations):
        if (i + 1) % 10000 == 0:
            prints = True
        else:
            prints = False
        predicted = exec_function(domain_radius, k_values)
        loss = calc_loss(predicted, y_values)
        if prints:
            print("\nLoss: " + str(loss))
        derivatives = []
        for num, key in enumerate(k_values.keys()):
            der = dl_dki(num, y_values, predicted, domain_radius)
            derivatives.insert(num, der)
            if prints:
                print(" - Derivative of k_" + str(num) + " = " + str(der))
        for num, key in enumerate(k_values.keys()):
            k_values[key] += -derivatives[num] * lr
    print("--FINISHED--")
    for num, key in enumerate(k_values.keys()):
        converted_k = k_values[key] / math.pow(domain_radius, num)
        k_values[key] = converted_k
        print("k_" + str(num) + ": " + str(k_values[key]))

## Get iterations, run, and give the user the original values too
## for comparison
iterations = int(input("Iterations?: "))
run(0.001, ks, ys, iterations)
print("Original values (k_0 to k_5): " + str(originals))
