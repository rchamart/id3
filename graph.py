import matplotlib.pyplot as plt

# x, y values where x = training percent, y = training accuracy, and z = test accuracy
x = [4, 10, 15, 18, 20, 30, 35, 40, 45, 50]

y = [79.9000, 85.2000, 87.1000, 88.1000, 88.7000, 90.5000, 91.1000, 91.3000, 91.3000, 91.4000]

z = [78.2000, 78.2000, 77.6000, 76.0000, 74.5000, 74.2000, 74.7000, 74.3000, 73.8000, 75.0000]

m = [75, 75, 75, 75, 75]

num_nodes = [447, 421, 439, 563, 477]


# x, y values where x = training percent and y = training accuracy
l1, = plt.plot(x , y, label = "% vs Training Set Accuracy")

plt.xlabel('% taken for Training')
plt.ylabel('% Accuracy')

# giving a title to my graph
plt.title('Pruned Tree Analysis: % Training vs Accuracy')

l2, = plt.plot(x , z, label = "% vs Testing Set Accuracy")

plt.legend()
plt.grid()
plt.show()