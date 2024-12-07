
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
from datetime import timedelta
from time import perf_counter
from pathlib import Path

# Constants
LEARNING_RATE = 0.01
THETA0 = 0
THETA1 = 0


# Data manipulation

def normalize(value: float, min: float, max: float) -> float:
    return (value - min) / (max - min)

def denormalize(value: float, min: float, max: float) -> float:
    return value * (max - min) + min

def normalize_data(data):
    distances = [pair[0] for pair in data]
    prices = [pair[1] for pair in data]

    min_km, max_km = min(distances), max(distances)
    min_price, max_price = min(prices), max(prices)

    return [(normalize(pair[0], min_km, max_km), normalize(pair[1], min_price, max_price)) for pair in data]

def denormalize_thetas(theta0, theta1, min_km, max_km, min_price, max_price) -> tuple[float, float]:
    new_theta0 = min_price + theta0 * (max_price - min_price) - theta1 * (min_km * (max_price - min_price) / (max_km - min_km))
    new_theta1 = theta1 * (max_price - min_price) / (max_km - min_km)

    return new_theta0, new_theta1

# Linear regression functions

def estimate_price(value: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * value

def calculate_theta0(data: list, theta0, theta1) -> float:
    summation = 0
    for pair in data:
        summation += estimate_price(pair[0], theta0, theta1) - pair[1]
    return LEARNING_RATE * (1 / len(data)) * summation

def calculate_theta1(data: list, theta0, theta1) -> float:
    summation = 0
    for pair in data:
        summation += (estimate_price(pair[0], theta0, theta1) - pair[1]) * pair[0]
    return LEARNING_RATE * (1 / len(data)) * summation

def train(data: list, iterations: int, theta0: float, theta1: float):
    ts = perf_counter()
    for _ in range(iterations):
        theta0 -= calculate_theta0(data, theta0, theta1)
        theta1 -= calculate_theta1(data, theta0, theta1)
    te = perf_counter()
    if args.time:
        print(f"Time needed for {args.iterations} iterations is {round((te - ts) * 1e3, 3)}ms")
    return theta0, theta1


# More "data science" functions

def calculate_r_squared(data: list[int|float], predicted_data: list[int|float]) -> None:
    prices_mean = sum(data) / len(data)
    residuals = [(price - predicted) ** 2 for price, predicted in zip(data, predicted_data)]

    ss_res = sum(residuals)
    ss_tot = sum([(price - prices_mean) ** 2 for price in data])

    r_squared = 1 - ss_res / ss_tot

    if not args.quiet:
        print(f"R\u00B2 = {r_squared}")

def calculate_mse(data: list[int|float], predicted_data: list[int|float]) -> None:
    residuals = [(price - predicted) ** 2 for price, predicted in zip(data, predicted_data)]
    mse = sum(residuals) / len(data)

    if not args.quiet:
        print(f"MSE = {mse}")

def calculate_mae(data: list[int|float], predicted_data: list[int|float]) -> None:
    residuals = [abs(price - predicted) for price, predicted in zip(data, predicted_data)]
    mae = sum(residuals) / len(data)

    if not args.quiet:
        print(f"MAE = {mae}")


# Utilities

def read_file(filename: str) -> list:
    try:
        with open(filename, 'r') as file:
            return [[int(num) for num in line.split(',')] for line in file.readlines()[1:]]
    except:
        if not args.quiet:
            print(f"Error: cannot read from '{filename}'")
        exit(1)

def save_file(filename: str, text: str) -> None:
    try:
        with open(filename, 'w') as file:
            file.write(text)
    except:
        if not args.quiet:
            print(f"Error: cannot write in '{filename}'")
        exit(1)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a simple model using linear regression.")

    parser.add_argument('dataset', type=Path, help="The path to the dataset used for training the model.")

    parser.add_argument('--plot-all', action='store_true', default=False, help="Plot all information (comprehends all options starting with '--plot').")
    parser.add_argument('--plot-data', action='store_true', default=False, help="Plot the dataset values.")
    parser.add_argument('--plot-model', action='store_true', default=False, help="Plot the model line.")
    parser.add_argument('--plot-legend', action='store_true', default=False, help="Shows the plot legend.")

    parser.add_argument('--r-squared', action='store_true', default=False, help="Calculate the Coefficient of Determination (R\u00B2).")
    parser.add_argument('--mse', action='store_true', default=False, help="Calculate the Mean Squared Error (MSE).")
    parser.add_argument('--mae', action='store_true', default=False, help="Calculate the Mean Absolute Error (MAE).")

    parser.add_argument('-i', '--iterations', type=int, default=10000, help="Specify the amount of iterations to train the model.")
    parser.add_argument('-o', '--output', help="Save the theta values into a file.")
    parser.add_argument('-t', '--time', action='store_true', default=False, help="Show the time spent to calculate the model.")

    parser.add_argument('-q', '--quiet', action='store_true', default=False, help="Do not print anything.")

    return parser.parse_args()





def main() -> None:
    global args
    args = parse_args()

    data = read_file(args.dataset)

    theta0, theta1 = train(normalize_data(data), args.iterations, THETA0, THETA1)

    distances = [pair[0] for pair in data]
    prices = [pair[1] for pair in data]

    theta0, theta1 = denormalize_thetas(theta0, theta1, min(distances), max(distances), min(prices), max(prices))

    # Print results
    if not args.quiet:
        print(f"\u03B8\u2080 = {theta0}\n\u03B8\u2081 = {theta1}")

    # Save to file
    if args.output is not None:
        save_file(args.output, f"{theta0}   {theta1}")

    predicted_prices = [estimate_price(pair[0], theta0, theta1) for pair in data]

    if args.r_squared:
        calculate_r_squared(prices, predicted_prices)

    if args.mse:
        calculate_mse(prices, predicted_prices)

    if args.mae:
        calculate_mae(prices, predicted_prices)

    if args.plot_data or args.plot_all:
        plt.plot(distances, prices, 'bo', label="dataset")

    if args.plot_model or args.plot_all:
        plt.plot(distances, predicted_prices, color='red', label="model")

    if args.plot_legend or args.plot_all:
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

