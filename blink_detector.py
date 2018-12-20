from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import time
import dlib
import imutils
import argparse
import cv2

class Blink():
	def __init__(self):
		self.eye_aspect_ratio_thresh = 0.3
		self.eye_aspect_ratio_consec_frames = 3
		self.counter = 0
		self.total = 0

	def eye_aspect_ratio(self, eye):
		'''
		compute the euclidean distances between the two sets of
		vertical eye landmarks coordinates
		'''
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		'''
		compute the euclidean distance between the horizontal
		eye landmark coordinates
		'''
		C = dist.euclidean(eye[0], eye[3])

		ratio = (A + B) / (2.0 * C)

		return ratio

	def start_detecting(self, detector, predictor, x, y):
		print('Starting Video Stream...')
		vs = VideoStream(src=0).start()
		(ls, le) = x
		(rs, re) = y
		while True:
			frame = vs.read()
			frame = imutils.resize(frame, width=450)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			rects = detector(gray, 0)

			for rect in rects:
				# determine the facial landmarks and convert into numpy array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# extract left and right eye coordinates
				left_eye = shape[ls : le]
				right_eye = shape[rs : re]

				# calculate the eye aspect ratio
				left_ratio = self.eye_aspect_ratio(left_eye)
				right_ratio = self.eye_aspect_ratio(right_eye)

				avg_ratio = (left_ratio + right_ratio) / 2.0

				#compute the convex hull for the eye
				left_hull = cv2.convexHull(left_eye)
				right_hull = cv2.convexHull(right_eye)

				# draw the contour
				cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

				# check that ratio is < blink threshold
				if avg_ratio < self.eye_aspect_ratio_thresh:
					self.counter += 1
				else:
					if self.counter > self.eye_aspect_ratio_consec_frames:
						self.total += 1

					# reset the counter
					self.counter = 0

				cv2.putText(frame, 'Blinks: {}'.format(self.total), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
				cv2.putText(frame, 'Ratio: {}'.format(avg_ratio), (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

			cv2.imshow('Frame', frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break

		cv2.destroyAllWindows()
		vs.stop()



def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--shape-predictor',
					help='path to facial landmark detector')
	args = vars(ap.parse_args())

	blink = Blink()

	print('Loading Detector...')
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args['shape_predictor'])

	'''
	grab the indexes of the facial landmarks of the left and
	right eye respectively
	'''
	(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
	(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

	blink.start_detecting(detector, predictor, (left_start, left_end), (right_start, right_end))

if __name__ == '__main__':
	main()


