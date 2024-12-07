
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
from pathlib import Path
from cmd import Cmd



def estimate_price(value: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * value

# Utility functions

def read_thetas(filename: str) -> tuple[float, float] | None:
    try:
        with open(filename, 'r') as file:
            return [float(n) for n in file.read().split()]
    except:
        print(f"Error: while reading thetas file '{filename}'")
        exit(1)

def get_thetas() -> tuple[float, float]:
    if args.input is not None:
        return read_thetas(args.input)
    if len(args.thetas) != 2:
        print(f"Error: you need to provide either two float numbers or a file path")
        exit(1)
    return args.thetas[0], args.thetas[1]

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a simple model using linear regression.")

    parser.add_argument('thetas', type=float, nargs='*', help="\u03B8\u2080 and \u03B8\u2081 values.")

    parser.add_argument('-i', '--input', type=Path, help="File containing \u03B8\u2080 and \u03B8\u2081 values.")

    return parser.parse_args()




class CommandPrompt(Cmd):
    intro = "Welcome! Type ? or help to list commands. Type exit to quit."
    prompt = "linear@regression:)$ "

    distances: list[int]
    prices: list[int]

    def __init__(self, completekey = "tab", stdin = None, stdout = None):
        super().__init__(completekey, stdin, stdout)
        self.theta0, self.theta1 = get_thetas()
        self.distances = []
        self.prices = []

    def do_predict(self, arg) -> None:
        '''
        Predict the price of a distance.
        Argument must be an integer greater than zero.
        '''
        try:
            distance = int(arg)
            if distance <= 0: raise Exception()
        except:
            print("Argument must be an integer and > 0")
            return

        price = int(estimate_price(distance, self.theta0, self.theta1))
        self.distances.append(distance)
        self.prices.append(price)

        print(f"Estimated price for {arg} is {price}")

    def do_show(self, arg):
        '''Show the number of predictions stored.'''
        print(f"Predictions made are {len(self.distances)}")

    def do_clear(self, arg) -> None:
        '''Delete all previously stored predictions.'''
        self.distances = []
        self.prices = []

    def do_plot(self, arg) -> None:
        '''Plot the stored predictions.'''
        if len(self.distances) <= 0:
            print("Not enough data to plot.")
            return
        plt.plot(self.distances, self.prices, 'o', color='blue')
        plt.plot(self.distances, self.prices, color='red', lw='0.5')
        plt.show()

    def do_exit(self, arg):
        '''Plot the stored predictions and exit.'''
        if arg != "":
            self.do_plot(arg)
        return True

    def do_help(self, arg):
        """List all available commands"""
        super().do_help(arg)

    def default(self, line):
        print(f"Unknown command: {line}. Type ? or help.")


def main() -> None:
    global args
    args = parse_args()

    CommandPrompt().cmdloop()

if __name__ == '__main__':
    main()
