import face_recognition
image = face_recognition.load_image_file("selfies.jpg")

face_locations = face_recognition.face_locations(image)
print(len(face_locations))

from matplotlib import pyplot as plt

print(plt.imshow(image))
plt.show()
