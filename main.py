from collections import defaultdict

from deepface import DeepFace as DF
import numpy as np
import shutil
import time
from matplotlib import pyplot as plt
import replicate
import os
import requests
import concurrent.futures
import firebase_admin

from firebase_admin import db, credentials, storage
import datetime

os.environ["REPLICATE_API_TOKEN"] = "r8_eXJyxamZbCUsiyR5qRsTcZ9CM34Cywc1w7e87"
images_path = "images_database_passport_size/"
detected_images_path = "detected/"
grp_images_path = "group_images/"

models = ["ArcFace", 'VGG-Face', "Facenet"]

all_combis = output = [[a, b] for a in models
                       for b in ['retinaface', 'mtcnn'] if a != b]

result = []
excluded = []


def calculate_all_enhanced(size):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=size)
    length = len(os.listdir("enhanced_images"))
    if not os.path.exists("enhanced_images"):
        os.mkdir("enhanced_images")
    if length != 0:
        shutil.rmtree("enhanced_images")
        os.mkdir("enhanced_images")
    for detected_image in os.listdir(detected_images_path):
        img_path = detected_images_path + detected_image
        pool.submit(calculate_enhanced_faces(img_path, detected_image))
    pool.shutdown(wait=True)


def recognize_faces():
    size = detect_faces()
    calculate_all_enhanced(size)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=size)
    if size != 0:
        for item in os.listdir(images_path):
            if item.endswith(".pkl"):
                item_path = images_path + item
                os.remove(item_path)
        for detected_img in os.listdir(detected_images_path):
            found_persons = []
            if detected_img not in excluded:
                img_path = detected_images_path + detected_img
                for combi in all_combis:
                    try:
                        pool.submit(found_persons.append(find_faces(combi, img_path)))
                    except:
                        continue
                    try:
                        for enhanced_img in os.listdir("enhanced_images"):
                            imgPath = "enhanced_images/" + enhanced_img
                            pool.submit(found_persons.append(find_faces(combi, imgPath)))
                    except:
                        continue
            if len(found_persons) != 0:
                unique_name_count = defaultdict(int)
                for name in found_persons:
                    unique_name_count[name] += 1
                    if unique_name_count[name] > 5:
                        excluded.append(name)
                result.append(sorted(unique_name_count.items(), key=lambda x: x[1], reverse=True)[0])
    pool.shutdown(wait=True)


def detect_faces():
    detected_images_folder_size = len(os.listdir(detected_images_path))
    for grp_img in os.listdir(grp_images_path):
        img_path = grp_images_path + grp_img
        faces1 = DF.extract_faces(img_path, target_size=(512, 512), detector_backend="retinafacez")
        #faces2 = DF.extract_faces(img_path, target_size=(512, 512), detector_backend="retinaface")
        if not os.path.exists("detected"):
            os.mkdir(detected_images_path)
        if detected_images_folder_size != 0:
            shutil.rmtree(detected_images_path)
            os.mkdir(detected_images_path)

        for i, img in enumerate(faces1):
            plt.imsave(f'detected/detected_image_{i}.png', faces1[i]['face'])

    return len(os.listdir(detected_images_path))


def find_faces(combi, img_path):
    df = DF.find(db_path=images_path, img_path=img_path, detector_backend=combi[1],
                 model_name=combi[0],
                 enforce_detection=False)
    df = np.array(df)
    ans = df[0][0][0]
    name = (str(ans).rsplit('/', 1)[1])
    return name


def calculate_enhanced_faces(img_path, image_name):
    model = replicate.models.get("tencentarc/gfpgan")
    version = model.versions.get("9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3")

    inputs = {
        'img': open(img_path, "rb"),
        'version': "v1.3",
        'scale': 2,
    }

    res_url = version.predict(**inputs)
    response = requests.get(res_url)
    res_path = "enhanced_images/" + "new_" + image_name
    with open(res_path, "wb") as f:
        f.write(response.content)
    return True


def updateFirebaseDB(subject_name):
    resultt = [("S1_AkshadC", 2), ("S2_GayatriM", 2), ("S3_VidyaJain", 2), ("S4_SagarM", 2), ("S5_MousamSingh", 2),
               ("S6_TejalP", 2), ("S7_SankalpW", 2), ("S8_MonikaN", 2), ("S9_ShreyasJ", 2), ("S10_LaluNair", 2)]

    cred = credentials.Certificate('beprojecttrial-firebase-adminsdk-pgvuf-96845622a4.json')
    firebase_admin.initialize_app(cred, name="MERI MARZI", options={
        'databaseURL': 'https://beprojecttrial-default-rtdb.firebaseio.com/'
    })

    ref = db.reference("Students/CE/BE/B", app=firebase_admin.get_app(name='MERI MARZI'))
    fileNames = [f for f in os.listdir("images_database_passport_size") if
                 os.path.isfile(os.path.join("images_database_passport_size", f))]
    filenames = ["S1_AkshadC", "S2_GayatriM", "S3_VidyaJain", "S4_SagarM", "S5_MousamSingh", "S6_TejalP",
                 "S7_SankalpW", "S8_MonikaN", "S9_ShreyasJ", "S10_LaluNair"]
    foundNames = []
    for n in resultt:
        foundNames.append(n[0])
    current_date = str("2023-06-10")
    for name in resultt:
        name = name[0]
        Sid = str(name).split('_', 1)[0]
        Sname = str(name).split('.', 1)[0]
        if Sid in ref.get():
            path = "Attendance/CE/BE/B" + "/" + Sname + "/" + current_date
            ref_init = db.reference(path, app=firebase_admin.get_app(name='MERI MARZI'))
            list_ref = ref_init.get()
            if list_ref is not None:
                list_ref.append(subject_name)
                ref_init.set(list_ref)
            else:
                ref_init.set([subject_name])

    for name in filenames:
        # name = name.split('.')[0]
        if name not in foundNames:
            Sid = str(name).split('_', 1)[0]
            Sname = str(name).split('.', 1)[0]
            if Sid in ref.get():
                path = "Attendance/CE/BE/B" + "/" + Sname + "/" + current_date
                ref_init = db.reference(path, app=firebase_admin.get_app(name='MERI MARZI'))
                list_ref = ref_init.get()
                if list_ref is not None:
                    res = "AB (" + subject_name + ")"
                    list_ref.append(res)
                    ref_init.set(list_ref)
                else:
                    res = "AB (" + subject_name + ")"
                    ref_init.set([res])


def download_images():
    cred = credentials.Certificate('beprojecttrial-firebase-adminsdk-pgvuf-96845622a4.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'beprojecttrial.appspot.com'
    })
    bucket = storage.bucket()
    pp_folder_path = "CE_BE_B/"
    gi_folder_path = "GroupImages/"
    blob1 = bucket.list_blobs(prefix=pp_folder_path)
    blob2 = bucket.list_blobs(prefix=gi_folder_path)

    if os.path.exists('group_images'):
        shutil.rmtree('group_images')
    if os.path.exists('images_database_passport_size'):
        shutil.rmtree('images_database_passport_size')
    list_b1 = list(blob1)
    list_b2 = list(blob2)
    size_pp_images = len(list_b1)
    size_grp_images = len(list_b2)

    for blob in range(1, size_pp_images):

        filename = os.path.basename(list_b1[blob].name)

        if not os.path.exists('images_database_passport_size'):
            os.makedirs('images_database_passport_size')

        file_path = f'images_database_passport_size/{filename}'
        list_b1[blob].download_to_filename(file_path)

    for blob in range(1, size_grp_images):
        if not os.path.exists('group_images'):
            os.makedirs('group_images')
        filename = os.path.basename(list_b2[blob].name)

        list_b2[blob].download_to_filename(f'group_images/{filename}')


def main():
    start = time.time()
    # download_images()
    #recognize_faces()
    updateFirebaseDB("HPC")
    if len(result) != 0:
        print(f"Students Found in the image for the given subject are: {result} ")

    else:
        print("No Students Found")
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
