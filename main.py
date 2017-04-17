import sys
import cv2
import numpy as np
from numpy import linalg
from skimage import color
from sklearn.cluster import DBSCAN

#-- Constants ------------------------------------

N = 50   # window size - 1

#-- Main ------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    sys.exit()

print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640))
print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480))

print("frame width = %s" % cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("frame height = %s" % cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def get_ssum(f):
    f = np.pad(f, (1,0), 'constant')
    csum = f.cumsum(0).cumsum(1)
    rsum = csum[N:,N:] - csum[N:,:-N] - csum[:-N,N:] + csum[:-N,:-N]
    ravg = rsum / (N**2)
    return(ravg)

def find_square(f):
    return(np.unravel_index(f.argmax(), f.shape))

global frame
frame = None

global lumination
lumination = None

global ab
ab = None

global reference_color
reference_color = np.array([0.17100391, 0.86445977, 0.93448082])

def on_mouse(e, x, y, f, p):
    global reference_color
    if e == cv2.EVENT_LBUTTONDOWN:
        reference_color = frame[y, x]

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', on_mouse)

global frame_count
frame_count = 0

global db
db = DBSCAN(eps=2.2, min_samples=10)

while(True):
    frame_count = frame_count + 1
    print('frame %s --------------------------------' % frame_count)

    # Capture the frame
    ret, raw_frame = cap.read()

    frame = color.rgb2lab(raw_frame)
    frame[:,:,0] = frame[:,:,0]/50
    frame[:,:,1:] = frame[:,:,1:]/80 + 0.5

    dist_frame = linalg.norm(frame - reference_color, axis=2)

    thresholded_dist_frame = cv2.inRange(dist_frame, 0, 0.3)
    active_pixel_locs = np.argwhere(thresholded_dist_frame)
    if len(active_pixel_locs) > 0:
        db.fit(active_pixel_locs)
        n_clusters = db.labels_.max() + 1
        print('num of samples = %s' % n_clusters)
        for i in range(n_clusters):
            #print('cluster %s (size = %s)' % (i, sum(db.labels_ == i)))
            if sum(db.labels_ == i) > 150:
                pixel_locs = active_pixel_locs[db.labels_ == i,:]
                maxs = pixel_locs.max(axis=0)
                mins = pixel_locs.min(axis=0)
                cv2.rectangle(raw_frame, (mins[1], mins[0]), (maxs[1], maxs[0]), (255,255,255), 2)
                cv2.rectangle(raw_frame, (mins[1], mins[0]), (maxs[1], maxs[0]), (0,0,0), 1)
    else:
        print('len(active_pixel_locs) = %d' % len(active_pixel_locs))

    print('reference_color = %s' % reference_color)


    #bin, contours, hierarchy = cv2.findContours(thresholded_dist_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours.shape)

    cv2.imshow('frame', raw_frame)
    cv2.moveWindow('frame', 1060, 520)
    #cv2.imshow('l', frame[:,:,0])
    #cv2.imshow('l', thresholded_dist_frame)
    #cv2.imshow('a', frame[:,:,1])
    #cv2.imshow('b', frame[:,:,2])

    #cv2.moveWindow('l', 400, 0)
    #cv2.moveWindow('a', 1060, 0)
    #cv2.moveWindow('b', 400, 520)

    # Looks like OpenCV insists using V4L which converts the YUYV format
    # to RGB internally, so no way to get better color resolution than
    # this 256-discretization.
    #layers = (frame / 255).transpose([2,0,1])

    #r = layers[2]
    #g = layers[1]
    #b = layers[0]

    ##total = get_ssum(r+g+b)

    #rq = get_ssum(r/(r+g+b+1)*2)
    #gq = get_ssum(g/(r+g+b+1)*2)
    #bq = get_ssum(b/(r+g+b+1)*2)

    #ry, rx = find_square(rq)
    #gy, gx = find_square(gq)
    #by, bx = find_square(bq)

    ##print('rc = %s, gc = %s, bc = %s' % (rc, gc, bc))
    ##print('%s, %s, %s' % (r[rc], g[gc], b[bc]))

    #if rq[ry, rx] > 0.65:
    #    cv2.rectangle(frame, (rx, ry), (rx+N, ry+N), (0,0,255), thickness=5)

    #if gq[gy, gx] > 0.65:
    #    cv2.rectangle(frame, (gx, gy), (gx+N, gy+N), (0,255,0), thickness=5)

    #if bq[by, bx] > 0.65:
    #    cv2.rectangle(frame, (bx, by), (bx+N, by+N), (255,0,0), thickness=5)

    ## Display the resulting frame
    #cv2.imshow('frame', frame)
    #cv2.imshow('r', rq)
    #cv2.imshow('g', gq)
    #cv2.imshow('b', bq)

    #cv2.moveWindow('frame', 660, 520)
    #cv2.moveWindow('r', 0, 0)
    #cv2.moveWindow('g', 660, 0)
    #cv2.moveWindow('b', 0, 520)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
