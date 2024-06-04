import matplotlib.pyplot as plt

# Epoch data
epochs = list(range(11))

# Corrected data for Adaboost
mean_loss_adaboost = [
     2.326117975954, 2.328264932552, 0.271376267102, 0.134331500812,
    0.079309606586, 0.066758197325, 0.042260469531, 0.039436030635, 0.026567832208,
    0.023397764162, 0.001654433780
]
test_acc_adaboost = [
    0.123700, 0.123700, 0.169900, 0.956800, 0.962000,
    0.976200, 0.975800, 0.976800, 0.976900, 0.979900,
    0.982400
]

# Create subplots for Mean Loss and Test Accuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot Mean Loss per Epoch
ax1.plot(epochs, mean_loss_adaboost, label='Adaboost', marker='o', color='blue')
ax1.set_title('Mean Loss per Epoch - Adaboost')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Mean Loss')
ax1.legend()

# Plot Test Accuracy per Epoch
ax2.plot(epochs, test_acc_adaboost, label='Adaboost', marker='o', color='red')
ax2.set_title('Test Accuracy per Epoch - Adaboost')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Accuracy')
ax2.legend()

# Adjust layout for clarity and show plot
plt.tight_layout()
plt.show()