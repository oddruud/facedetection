from deepface import DeepFace
import json
import numpy as np

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

#def create_embedding_nearest_neighbors_database():
    


def test_face_detect_models():
    test_img_path = "data/test.jpg"
    test_img_path2 = "data/test2.jpg"
    natalie_test_img = "data/natalie.png"

    database = "data/database"

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


def detect_face(pillow_image):
    model = "VGG-Face"
    backend = "yunet"
    metric = "cosine"
    database = "data/database"
    
    person_id = "unknown"
    bounding_box = {"x":0, "y":0, "width":0, "height":0}
    
    #spotify annoy
    image_numpy = np.array(pillow_image)
    width = float(image_numpy.shape[1])
    height = float(image_numpy.shape[0])
    
    print("width: ", width, ", height: ", height)
    
    #model:  Facenet , backend:  yunet , distance metric:  euclidean_l2
    dfs = DeepFace.find(img_path = image_numpy, 
            db_path = database, 
            model_name= model,
            distance_metric = metric,
            enforce_detection=False,
            detector_backend=backend
            )
    
    image_to_person_lookup = json.loads(open("image_to_person_lookup.json").read())
    
   # print("image to person lookup json: " + image_to_person_lookup) 
    if dfs is None:
            return person_id, bounding_box
    
    if len(dfs) == 0:
        return person_id, bounding_box
    
    if len(dfs[0].identity) == 0:
        return person_id, bounding_box
    
    print("dfs: ", dfs)
    identity = dfs[0].identity[0]
    
    for people in image_to_person_lookup:
        if len(dfs) > 0 and people["photo_id"] in identity:
            person_id = people["name"]
            #normalized
            bounding_box["x"] = float(dfs[0].source_x[0]) / width
            bounding_box["y"] = float(dfs[0].source_y[0]) / height
            bounding_box["width"] = float(dfs[0].source_w[0]) / width
            bounding_box["height"] = float(dfs[0].source_h[0]) / height
            
            return person_id, bounding_box
    
    return person_id, bounding_box