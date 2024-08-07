import matplotlib.pyplot as plt

# Data for the bars
labels = ['CINIC', 'IMAGENETTE', 'EMNIST']
values = [5*(1-0.891-0.023), 5 * 0.023, 5 * 0.891 ]

# Creating the bar plot
fig, ax = plt.subplots()
bars = ax.bar(labels, values, color=['blue', 'green', 'red'])

# Adding title and text annotation
ax.set_title('Hierarchy Target Domain = MNIST')
ax.text(1.5, max(values)+0.5 , f'Estimated domain size = {5}k', ha='center', va='bottom', fontsize=12, color='black')

# Adding value annotations on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Adjusting y-axis limit for better visualization
plt.ylim(0, max(values) + 2)

# Saving the plot to a file
plt.savefig('Hierarchy_MNIST.png')

# Displaying the plot
plt.show()
