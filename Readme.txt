This is an end to end deep learning project series in agriculture domain. Farmers every year face economic loss and crop waste due to various diseases in potato plants. I will use image classification using CNN and built a mobile app using which a farmer can take a picture and app will tell you if the plant has a disease or not. 

Dataset link : 
https://www.kaggle.com/arjuntejaswi/plant-village


Technology stack for this project are : 

1. Model Building: tensorflow, CNN, data augmentation, tf dataset

2.Backend Server and ML Ops: tf serving, FastAPI

3.Model Optimization: Quantization, Tensorflow lite

4.Frontend: React JS

5.Deployment: GCP (Google cloud platform, GCF (Google cloud functions)



Using FastAPI & TF Serve

1. Get inside api folder
cd api

2. Copy the models.config.example as models.config and update the paths in file.

3. Run the TF Serve (Update config file path below)

docker run -t --rm -p 8501:8501 -v C:/Code/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease-classification/models.config



Deploying the TF on GCP : 

1. Create a GCP account.

2. Create a Project on GCP (Keep note of the project id).

3. Create a GCP bucket.

4. Upload the model.h5 model in the bucket in the path models/model.h5.

5. Install Google Cloud SDK (Setup instructions).

6. Authenticate with Google Cloud SDK.
gcloud auth login

7. Run the deployment script.
cd gcp
gcloud functions deploy predict_lite --runtime python38 --trigger-http --memory 512 --project project_id


Inspiration: How to serve deep learning models using TensorFlow 2.0 with Cloud :

Functionshttps://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions
