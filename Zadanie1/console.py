from objects.pyramid import print_pyramid
from objects.rectangle import print_rectangle
from objects.star import print_star

from objects.square import print_square


def pyramid(size):
    print_pyramid(size)


def square(size):
    print_square(size)


def rectangle(size):
    print_rectangle(size)


def star(size):
    print_star(size)


def main():
    pattern = ""
    while pattern != "!x":
        pattern = input("Type !help for more info, !x for exit: ")

        if pattern == "!help":
            print("Printable object are square, pyramid and star")
        elif pattern == "!x":
            return
        else:
            if pattern == "square":
                size = int(input("size:"))
                square(size)
            if pattern == "rectangle":
                size = int(input("size:"))
                rectangle(size)
            elif pattern == "pyramid":
                size = int(input("size:"))
                pyramid(size)
            elif pattern == "star":
                size = int(input("size:"))
                star(size)
            else:
                print("unknown command")


if __name__ == "__main__":
    main()
