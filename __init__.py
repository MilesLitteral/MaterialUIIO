# marketplace/marketplace.py
import os
import grpc
from   flask                      import  Flask,  render_template
from   supabase                   import  Client, create_client
from   TextureDefinition_pb2_grpc import  RecommendationsStub
from   TextureDefinition_pb2      import  TextureCategory, RecommendationRequest

import json
import time 
import numpy
import pillow
import requests
import tensorflow     as tf
import tensorflow_hub as hub
import pyngrok        as ngrok
from   textureGAN import get_network, CONFIG
# requirements.txt 
# Pillow==9.1.1
# tensorflow
# tensorflow_hub
# requests
# numpy

# Open a SSH tunnel
# <NgrokTunnel: "tcp://foo.tcp.ngrok.io:12345" -> "localhost:22">
app              = Flask(__name__, template_folder='template')
url: str         = "https://yfyozxjziplqwwddnhgm.supabase.co" #os.environ.get() #SUPABASE_URL
key: str         = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlmeW96eGp6aXBscXd3ZGRuaGdtIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTIyOTkyODYsImV4cCI6MjAwNzg3NTI4Nn0.xxXJyikXVLt6MFubGmKfDxxiUppAavoPPprV2u9bhr0" #os.environ.get() #SUPABASE_KEY
supabase: Client = create_client(url, key)
signInUser       = None
ssh_tunnel = ngrok.connect(22, "tcp") #auth="username:password"
print(f"ADAPI SSH Tunnel: {ssh_tunnel.public_url} t")

recommendations_host    = os.getenv("RECOMMENDATIONS_HOST", "localhost")
recommendations_channel = grpc.insecure_channel(f"{recommendations_host}:50051")
recommendations_client  = RecommendationsStub(recommendations_channel)

IMAGE_SHAPE = (224, 224)
URL         = "http://localhost:9001/v1/models/1657646009:predict"
scale_layer = tf.keras.layers.Rescaling(1./255)

def load_model():
    classifier_model = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))
    ])
    return classifier

def preprocess(images, labels=None):
  images = tf.image.resize(scale_layer(images),[120, 120], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return images, labels

def save_model():
    model = load_model()
    ts    = int(time.time())
    base_file_path = os.getcwd() + "/models/"

    file_path = base_file_path + str(ts)
    model.save(filepath=file_path, save_format='tf')

def serve_rest(img_url: str):
    # image preprocessing using TF
    img     = preprocess(img_url)
    data    = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
    headers = {"content-type": "application/json"}
    
    # Sending request to the URL where the model is being served
    json_response = requests.post(URL, data=data, headers=headers)
    predictions   = json.loads(json_response.text)['predictions']
    
    # decoding the results of predictions
    results = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=5) #decode_predictions(predictions)
    
    if len(os.sys.argv) < 2:
        print('Usage:')
        print('  python %s network_name [num_iterations]\t- Trains a network on the images in the input folder'%os.sys.argv[0])
    elif len(os.sys.argv) < 3:
        get_network(os.sys.argv[1], **CONFIG).train()
    else:
        get_network(os.sys.argv[1], **CONFIG).train(int(os.sys.argv[2]))
    
    # returns a dictionary containing the class name 
    # and the probability with which it is predicted
    return results

#python -m pip install -r requirements.txt
@app.route("/", methods=['GET', 'POST'])
def render_homepage():
    data = supabase.auth.get_session()
    recommendations_request  = RecommendationRequest(user_id=1, category=TextureCategory.COLOR, max_results=3)
    recommendations_response = recommendations_client.Recommend(recommendations_request)
    return render_template(
        "index.html" ,
        recommendations=recommendations_response.recommendations,
    )

@app.route("/about", methods=['GET', 'POST'])
def render_about():
    data = supabase.auth.get_session()
    return render_template("about.html")

@app.route("/saas", methods=['GET', 'POST'])
def render_saas():
    data = supabase.auth.get_session()
    recommendations_request  = RecommendationRequest(user_id=1, category=TextureCategory.COLOR, max_results=3)
    recommendations_response = recommendations_client.Recommend(recommendations_request)
    return render_template(
        "saas.html" ,
        recommendations=recommendations_response.recommendations,
    )

@app.route("/register", methods=['POST'])
def register_user(email, password):
   supabase.auth.sign_up({
    "email":    email,
    "password": password,
    })
   
@app.route("/login", methods=['POST'])
def login_user(email, password):
    supabase.auth.sign_in_with_password({"email": email, "password": password})

@app.route("/logout", methods=['POST'])
def logout_user():
    sess = supabase.auth.get_session()
    supabase.auth.refresh_session(sess.access_token)
    supabase.auth.sign_out()

@app.route("/fetch_session")
def fetch_session():
    return supabase.auth.get_session()

@app.route("/fetch")
def fetch_data():
    response = supabase.table('textures').select("*").execute()

@app.route("/insert")
def insert_data():
    data, count = supabase.table('textures').insert({"id": 0, "name": "Denmark", "bytes": "NULL/x0"}).execute()

@app.route("/upload")
def render_upload(bucket_name, destination, source):
    data = supabase.auth.get_session()
    with open(source, 'wb+') as f:
      res = supabase.storage.from_(bucket_name).upload(destination, f)
    return res

@app.route("/download")
def render_download(bucket_name, destination, source):
    data = supabase.auth.get_session()
    with open(destination, 'wb+') as f:
      res = supabase.storage.from_(bucket_name).download(source)
      f.write(res)

@app.route("/download_url")
def render_download_url(bucket_name):
    data  = supabase.auth.get_session()
    res   = supabase.storage.from_(bucket_name).get_public_url('test/avatar1.jpg')
    return res

if __name__ == '__main__':
    app.run(debug=True)