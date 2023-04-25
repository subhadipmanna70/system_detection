import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
import pickle
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img, img_to_array
app = FastAPI()


class test(BaseModel):
    preg : int
    glucose : int
    bp : int
    st : int
    insulin : int
    bmi : float
    dpf : float
    age : int



rf_model = pickle.load(open('rf_model.pkl', 'rb'))

### diabetes api
@app.post('/diabetes_test')
async def testing(it: test):
    data = np.array([[it.preg, it.glucose, it.bp, it.st, it.insulin, it.bmi, it.dpf, it.age]])

    result = rf_model.predict(data)
    print(result)

    return {"prediction":int(result)}


##pneumonia api
@app.post('/pneumonia_test')
def create_upload_file(file: UploadFile = File(...)):
    data=file.file.read()

    with open("images/"+file.filename,"wb") as f:
        f.write(data)
    file.file.close()

    model = load_model('our_model.h5')  # Loading our model
    img = load_img(f"images/{file.filename}",
                         target_size=(224, 224))
    imagee = img_to_array(img)  # Converting the X-Ray into pixels
    imagee = np.expand_dims(imagee, axis=0)
    img_data = preprocess_input(imagee)
    prediction = model.predict(img_data)
    # os.remove(f"images/{file.filename}")
    if prediction[0][0] > prediction[0][1]:  # Printing the prediction of model.
        value = 0                     ### not affected
    else:
        value = 1                    ## affected
    return {"prediction": value}
