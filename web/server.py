from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from keras.models import load_model
import redis, sys, os, json, copy, time
[sys.path.append(i) for i in ['.', '..']]
import numpy as np

# app
app = Flask(__name__, static_url_path='')
CORS(app)

# redis
r = redis.Redis(host='127.0.0.1', port=6379) # redis instance host / port

# tf / keras model
encoder = load_model('../pose-enc-raymond.h5')
encoder._make_predict_function() # see https://github.com/keras-team/keras/issues/6462#issuecomment-319232504

# label order expected by model in input data
default_labels = [
  'C7', 'CLAV', 'LANK', 'LBHD', 'LBSH',
  'LBWT', 'LELB', 'LFHD', 'LFRM', 'LFSH',
  'LFWT', 'LHEL', 'LIEL', 'LIHAND', 'LIWR',
  'LKNE', 'LKNI', 'LMT1', 'LMT5', 'LOHAND',
  'LOWR', 'LSHN', 'LTHI', 'LTOE', 'LUPA',
  'MBWT', 'MFWT', 'RANK', 'RBHD', 'RBSH',
  'RBWT', 'RELB', 'RFHD', 'RFRM', 'RFSH',
  'RFWT', 'RHEL', 'RIEL', 'RIHAND', 'RIWR',
  'RKNE', 'RKNI', 'RMT1', 'RMT5', 'ROHAND',
  'ROWR', 'RSHN', 'RTHI', 'RTOE', 'RUPA',
  'STRN', 'T10'
]


def parse_frame(frame):
  '''
  Read in a frame with the following attributes:
    frame = {
      'coords': '-0.107206,-0.330513,0.440121,-0.0794874,-0.28018,0.401152,-0.306961,-0.339731,0.0579791,-0.0944004,-0.339983,0.478491,-0.111611,-0.303003,0.463764,-0.186795,-0.30791,0.31768,-0.0805198,-0.247602,0.512007,-0.0735331,-0.315659,0.500194,-0.0570994,-0.237104,0.534673,-0.0877471,-0.281142,0.438769,-0.183399,-0.264388,0.294658,-0.242364,-0.349816,0.0407568,-0.0573425,-0.239129,0.506003,0.0169797,-0.247543,0.587483,0.00253677,-0.246041,0.561359,-0.235924,-0.293908,0.189106,-0.214729,-0.315368,0.154069,-0.281331,-0.320805,0.012452,-0.30871,-0.317281,0.0132234,-0.000823545,-0.231602,0.594314,-0.014585,-0.229738,0.577916,-0.263557,-0.309513,0.116388,-0.215748,-0.284479,0.247467,-0.29102,-0.309049,0.00789521,-0.095618,-0.294208,0.474638,-0.159668,-0.341823,0.301349,-0.134224,-0.261114,0.292483,-0.0911074,-0.318012,0.0421214,-0.0565226,-0.355773,0.459094,-0.0859622,-0.35031,0.41992,-0.106736,-0.337805,0.290566,-0.0254738,-0.353885,0.41608,-0.0361978,-0.332534,0.478792,0.0181221,-0.351129,0.419029,-0.0690738,-0.312787,0.410748,-0.104676,-0.291026,0.28428,-0.118778,-0.321294,0.0226481,0.00231677,-0.328574,0.402622,0.0807949,-0.321145,0.473495,0.0628888,-0.323059,0.45939,-0.0925277,-0.314079,0.175116,-0.127643,-0.302796,0.152506,-0.0853917,-0.266112,0.0133026,-0.069451,-0.291896,0.0138023,0.0976319,-0.332964,0.460101,0.0811,-0.335654,0.440515,-0.0919298,-0.327819,0.0872194,-0.0894561,-0.322675,0.213304,-0.0662387,-0.266265,0.0197155,-0.0695913,-0.344314,0.421299,-0.104043,-0.253498,0.35348,-0.141684,-0.318906,0.38162',
      'frame_number': '2463',
      'markers': 'C7,CLAV,LANK,LBHD,LBSH,LBWT,LELB,LFHD,LFRM,LFSH,LFWT,LHEL,LIEL,LIHAND,LIWR,LKNE,LKNI,LMT1,LMT5,LOHAND,LOWR,LSHN,LTHI,LTOE,LUPA,MBWT,MFWT,RANK,RBHD,RBSH,RBWT,RELB,RFHD,RFRM,RFSH,RFWT,RHEL,RIEL,RIHAND,RIWR,RKNE,RKNI,RMT1,RMT5,ROHAND,ROWR,RSHN,RTHI,RTOE,RUPA,STRN,T10',
      'n_subjects': '1',
      'subject_id': '0',
      'subject_name': 'Raymond',
    }
  and return that frame with typed attributes.

  @args dict frame:
    frame with key strings and vals as identified above
  @returns dict:
    see `encode` input for k,v types
  '''
  frame['coords'] = np.float_(frame['coords'].split(',')).reshape((-1,3))
  frame['frame_number'] = int(frame['frame_number'])
  frame['markers'] = frame['markers'].split(',')
  frame['n_subjects'] = int(frame['n_subjects'])
  frame['subject_id'] = int(frame['subject_id'])
  return frame


def encode(frame, float_precision=3):
  '''
  Encode an incoming frame and return the latent space and orientation coords
  to the calling agent.

  @args dict frame
    dict with attributes:
      coords: str
        has an array of 156 float values separated by ,
        indicates the positional coordinates of a model in a single frame
      frame_number: str version of int the current frame index
      markers: str an array of 52 markers separated by ,
        indicates the order of the coords within this frame
      n_subjects: str version of int that indicates number of subjects in frame
      subject_id: str version of int that indicates the subject id for the coords
        in this packet
  @args int float_precision
    the number of decimal places to include in data returned from this function

  @returns dict
    a dictionary with the attributes
      'x': the raw positional coordinates of this frame; shape = (52,3)
      'z': the latent space coords for this frame; shape = (1,3)
      'w': the orientation of the model within this frame; shape = (1,2)
  '''
  frame = parse_frame(frame)
  ordering = [default_labels.index(i) for i in frame['markers']]
  # x is array with shape (52,3) with one point per vertex in `default labels`
  x = frame['coords'][ordering]
  # center the array
  x[:,:2] -= x[:,:2].mean(axis=-2, keepdims=True)
  # convert the input array into the right input shape
  expanded = np.expand_dims(x[ordering], axis=0)
  # find the position of this array in the latent space
  z, w = encoder.predict(expanded)
  return {
    'frame_number': frame.get('frame_number', ''),
    'x': np.around(x, float_precision).tolist(),
    'z': np.around(z, float_precision).tolist(),
    'w': np.around(w, float_precision).tolist(),
  }


##
# Routes
##

@app.route('/api/frame')
def get_frame():
  # fetch the latest frame from redis
  frame = json.loads(r.get('frame').decode('utf8'))
  return jsonify(encode(frame))


@app.route('/')
def index():
  # return the html content
  return send_from_directory('.', 'index.html')


##
# Main
##

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5050)