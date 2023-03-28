def print_star(size):
    for i in range(size):
        for j in range(size):
            if i == j or i == size-j-1 or i == size//2 or j == size//2:
                print("*", end="")
            else:
                print(" ", end="")
        print()