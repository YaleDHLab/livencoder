# livencoder
autoencoder visualization for live mocap data

## Requirements

This code must be run with python 3.7.3. To create a python 3.7.3 environment, install anaconda then run:

```
conda create -n 3.7.3 python=3.7.3
conda activate 3.7.3
```
Then pip install the required modules:

```
pip install -r utils/requirements.txt
```

## Matplotlib Usage

Pipe the emulator into the live plot utility to playback a pre-recorded mocap session.

```
./vicon-emulate.py | ./vicon-liveplot.py
```

## WebGL Usage

To stream the emulation data into a redis instance running on localhost:6379, run:
```
python vicon-emulate.py | python web/publisher.py
```

Then start the server that works as an api between the redis store and the web client:
```
FLASK_APP=server.py FLASK_DEBUG=1 flask run --port=5050
```