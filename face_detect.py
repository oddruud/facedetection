from deepface import DeepFace
import json

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace",
  "SFace",
]

backends = [
  'opencv', 
  'ssd', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

metrics = ["cosine", "euclidean", "euclidean_l2"]



def test_face_detect_models():
    test_img_path = "test_data/test.jpg"
    test_img_path2 = "test_data/test2.jpg"
    natalie_test_img = "test_data/natalie.png"

    database = "test_data/database"

    for model in models:
        for backend in backends:
            for metric in metrics:
                dfs = DeepFace.find(img_path = natalie_test_img, 
                db_path = database, 
                model_name= model,
                distance_metric = metric,
                enforce_detection=True,
                detector_backend=backend
                )
                print("model: ", model, ", backend: ", backend, ", distance metric: ", metric)
                print("FIND RESULT", dfs)


def detect_face(image_path):
    model = "VGG-Face"
    backend = "yunet"
    metric = "cosine"
    database = "test_data/database"
    
    #spotify annoy
    
    #model:  Facenet , backend:  yunet , distance metric:  euclidean_l2
    dfs = DeepFace.find(img_path = image_path, 
            db_path = database, 
            model_name= model,
            distance_metric = metric,
            enforce_detection=False,
            detector_backend=backend
            )
    
    image_to_person_lookup = json.loads(open("image_to_person_lookup.json").read())
    
   # print("image to person lookup json: " + image_to_person_lookup) 
    if dfs is None:
            return "Unknown"
    
    if len(dfs) == 0:
        return "Unknown"
    
    if len(dfs[0].identity) == 0:
        return "Unknown"
    
    print("dfs: ", dfs)
    identity = dfs[0].identity[0]
    for people in image_to_person_lookup:
        if len(dfs) > 0 and people["photo_id"] in identity:
            return people["name"]
    
    return "Unknown"