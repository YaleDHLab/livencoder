import sys, redis, json

# config
r = redis.Redis(host='127.0.0.1', port=6379) # redis instance host / port

print(' * publishing to redis')

# parse stream
frame = []
while True:
  l = sys.stdin.readline()
  if 'end_frame' in l:
    # add to the frames and trim if we have a surplus
    new_frame = {i.split(':')[0]: i.split(':')[1] for i in frame if ':' in i}
    r.set('frame', json.dumps(new_frame))
    frame = []
    #frame_number = print(' * published frame', new_frame.get('frame_number', ''))
  elif l.rstrip('\n'):
    frame.append(l.rstrip('\n'))