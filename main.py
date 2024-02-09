from deepface import DeepFace

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

test_img_path = "test_data/test.jpg"
database = "test_data/database"

metrics = ["cosine", "euclidean", "euclidean_l2"]

#face verification
result = DeepFace.verify(img1_path = test_img_path, 
          img2_path = "img2.jpg", 
          distance_metric = metrics[1]
)

print("VERIFY RESULT: ", result)

#face recognition
dfs = DeepFace.find(img_path = test_img_path, 
          db_path = database, 
          distance_metric = metrics[2]
)

print("FIND RESULT: ", result)

#face detection and alignment
face_objs = DeepFace.extract_faces(img_path = test_img_path, 
        target_size = (224, 224), 
        detector_backend = backends[4]
)

print("Face objects: ", face_objs)



