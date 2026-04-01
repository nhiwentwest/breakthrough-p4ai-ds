import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mnist_datasets import MNISTLoader
loader = MNISTLoader()
images, labels = loader.load()
assert len(images) == 60000 and len(labels) == 60000

# Load test dataset
test_images, test_labels = loader.load(train=False)
assert len(test_images) == 10000 and len(test_labels) == 10000


# Check dataset properties
print("Train images:", len(images))
print("Train labels:", len(labels))

print("Test images:", len(test_images))
print("Test labels:", len(test_labels))

# Reshape vector images

images_np = np.array(images)
images_reshaped = images_np.reshape(-1, 28, 28)
labels_np = np.array(labels)

print(images_reshaped.shape)

# Print sample images

plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images_reshaped[i], cmap="gray")
    plt.title(labels[i])
    plt.axis("off")

plt.suptitle("Some sample images")
plt.show()

#Check Digit distribution
sns.countplot(x=labels)
plt.title("Digit Distribution")
plt.xlabel("Digit")
plt.ylabel("Count")

plt.show()

#Check image value
images_np = np.array(images)

print("Min pixel value:", images_np.min())
print("Max pixel value:", images_np.max())


#Show examples for each digit
plt.figure(figsize=(10,5))

for digit in range(10):
    idx = np.where(labels == digit)[0][0]   # find first occurrence
    plt.subplot(2,5,digit+1)
    plt.imshow(images_reshaped[idx], cmap="gray")
    plt.title(f"Digit {digit}")
    plt.axis("off")

plt.suptitle("Example for each digit")
plt.show()

# Mean per digit
plt.figure(figsize=(12,5))

for digit in range(10):

    digit_images = images_reshaped[labels_np == digit]
    mean_image = digit_images.mean(axis=0)

    plt.subplot(2,5,digit+1)
    plt.imshow(mean_image, cmap="inferno")
    plt.title(f"Digit {digit}")
    plt.axis("off")

plt.suptitle("Mean Image per Digit")
plt.show()



