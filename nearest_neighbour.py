import math
import matplotlib.pyplot as plt


def nn(dataset, p):
    dist = []
    data0 = len(dataset[0])
    data1 = len(dataset[1])
    info0 = 0
    info1 = 0

    for cluster in dataset:
        for parameter in dataset[cluster]:
            euclidean_dist = math.sqrt((parameter[0] - p[0]) ** 2 + (parameter[1] - p[1]) ** 2)
            if cluster == 0:
                info0 += euclidean_dist
            elif cluster == 1:
                info1 += euclidean_dist
            dist.append((euclidean_dist, cluster))

    avg0 = info0 / data0
    avg1 = info1 / data1

    if avg0 > avg1:
        dataset[1].append(p)
    else:
        dataset[0].append(p)

    dist = sorted(dist)
    freq0 = 0
    freq1 = 0
    for d in dist:
        if d[1] == 0:
            freq0 += 1
        elif d[1] == 1:
            freq1 += 1

    if avg1 > avg0 and freq0 > freq1:
        return 0
    elif avg1 > avg0 and freq0 < freq1:
        print('Rogue Pattern')
        return 0
    elif avg1 < avg0 and freq0 < freq1:
        return 1
    elif avg1 < avg0 and freq0 > freq1:
        print("Rogue Pattern")
        return 1


def main():
    dataset = {0: [(1, 8), (2, 5), (2.3, 5), (1.3, 6), (1.5, 8), (2, 6), (2.1, 8), (1.2, 7), (2.2, 5.3), (1.8, 6)],
               1: [(5.5, 3), (6, 2), (6.2, 3), (5.7, 1), (6.4, 2), (5.8, 2), (5.6, 4.1), (5.7, 2), (6, 3), (5, 1.9)]}

    # p= 4 7

    while True:

        print('Enter single co-ordinate of x and y axes with spaces : ')
        p = [float(x) for x in input().split()]
        plt.figure(1)

        if nn(dataset, p) == 1:
            dataset[1].append(p)
        else:
            dataset[0].append(p)

        for cluster in dataset:
            for parameter in dataset[cluster]:
                if cluster == 0:
                    plt.plot(parameter[0], parameter[1], 'rx')
                elif cluster == 1:
                    plt.plot(parameter[0], parameter[1], 'bo')
        plt.xlabel('height')
        plt.ylabel('weight')
        plt.show()


if __name__ == "__main__":
    main()