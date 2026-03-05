
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2


# load our serialized model from disk
print("	- loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream and allow the cammera sensor to warmup
print("	- starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# tracking / counting state
next_face_id = 1
tracks = {}  # face_id -> {"centroid": (cx, cy), "bbox": (startX, startY, endX, endY), "lost": int}
unique_faces_count = 0

MAX_LOST_FRAMES = 480  # how many frames a track can disappear before we forget it
DISTANCE_THRESHOLD = 100  # max distance in pixels to match detections to existing tracks

print("	- Program is now running..")
print("	- Counting Attendees..")

print('''
============================================
	Press 'Q' to End the Program
============================================
''')
def _centroid_from_bbox(bbox):
	(startX, startY, endX, endY) = bbox
	cx = (startX + endX) // 2
	cy = (startY + endY) // 2
	return (cx, cy)


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# collect all strong detections for this frame
	current_bboxes = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence < 0.5:
			continue

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		current_bboxes.append((startX, startY, endX, endY))

	# associate detections with existing tracks using nearest-centroid matching
	updated_tracks = {}
	used_detection_indices = set()

	for face_id, track in tracks.items():
		prev_cx, prev_cy = track["centroid"]
		best_index = None
		best_dist = None

		for idx, bbox in enumerate(current_bboxes):
			if idx in used_detection_indices:
				continue

			cx, cy = _centroid_from_bbox(bbox)
			dist = np.hypot(cx - prev_cx, cy - prev_cy)

			if best_dist is None or dist < best_dist:
				best_dist = dist
				best_index = idx

		if best_index is not None and best_dist is not None and best_dist <= DISTANCE_THRESHOLD:
			bbox = current_bboxes[best_index]
			cx, cy = _centroid_from_bbox(bbox)
			updated_tracks[face_id] = {
				"centroid": (cx, cy),
				"bbox": bbox,
				"lost": 0,
			}
			used_detection_indices.add(best_index)
		else:
			# no good match this frame; keep the track for a while
			track["lost"] += 1
			if track["lost"] <= MAX_LOST_FRAMES:
				updated_tracks[face_id] = track

	# any remaining detections that weren't matched start new tracks
	for idx, bbox in enumerate(current_bboxes):
		if idx in used_detection_indices:
			continue

		cx, cy = _centroid_from_bbox(bbox)
		updated_tracks[next_face_id] = {
			"centroid": (cx, cy),
			"bbox": bbox,
			"lost": 0,
		}
		unique_faces_count += 1
		next_face_id += 1

	tracks = updated_tracks

	# draw everything
	current_faces_in_frame = len(tracks)

	for face_id, track in tracks.items():
		(startX, startY, endX, endY) = track["bbox"]

		# draw bounding box
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

		# draw ID label above the box
		label_y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(
			frame,
			f"ID {face_id}",
			(startX, label_y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 255, 0),
			2,
		)

	# overlay live counters
	cv2.putText(
		frame,
		f"Current faces: {current_faces_in_frame}",
		(10, 20),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(255, 255, 255),
		2,
	)
	cv2.putText(
		frame,
		f"Unique faces (session): {unique_faces_count}",
		(10, 45),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(255, 255, 255),
		2,
	)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break



cv2.destroyAllWindows()
vs.stop()

print(f'''Total Attendees Counted: {unique_faces_count}

''')
