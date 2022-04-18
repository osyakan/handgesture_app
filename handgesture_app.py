#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ハンドジェスチャによるアプリケーション
ハンドジェスチャによってスライダーを操作し、ターゲットに一致させる（ただひたすら）
４種類のハンドジェスチャを用意してある

<キー入力について説明>
a: ストレッチジェスチャに変更(両手間距離の伸縮)
s: ダイヤルジェスチャに変更(片手の角度)
d: 垂直スライドジェスチャに変更(片手の垂直方向移動)
f: 水平スライドジェスチャに変更(片手の水平方向移動)
v: GUI上の視覚フィードバックの表示/非表示をスイッチする
"""
import PySimpleGUI as sg
import time
import pyautogui as pag
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import random
import math

def main():
	# 引数解析 #################################################################
	cap_device = 0		#使用するカメラのid(confirm_camera.pyで確認)
	cap_width = 1280	#使用するカメラの横方向の画素
	cap_height = 720	#使用するカメラの縦方向の画素
	use_brect = True	

	#変数宣言 ##################################################################
	gesture_list = [0, 0, 0, 0]	#ジェスチャ処理関数にて値を受け取るリスト
	gesture_flag = 0			#ジェスチャの判別
	tmp_v = [0,0,0] 
	tmp_flag = False
	visible_flag = False		#視覚フィードバックの有無を判定(true->有)
	postponement = 4			#猶予フレーム数
	post_count = 0				#ジェスチャ入力が終了してから経過したフレーム数
	max_angle = 90				#ダイヤルジェスチャの最大可動域
	input_range = 100 			#値域(0~input_range)
	accept_error = min(int(input_range*0.05), 3) #誤差許容範囲
	sholder_length = 0.0		#肩の長さ
	pose_interval = -10			#最新の肩の長さを図った時間
	pose_cor = [(0,0), (0,0)]	

	#GUI系変数宣言
	sg.theme('BrownBlue')
	screen_w, screen_h = pag.size()
	radio_font=('Courier', int(screen_h/40))
	text_size = (('Courier', int(screen_h/7)), ('Courier', int(screen_h/12)))
	image_size = (640,360) #逆になる
	color_code = '#ffa500'
	default_color_code = '#ffffff'
	displaynext_color_code = '#64778d'
	embed_color_code = '#64778d'

	# カメラ準備 ###############################################################
	cap = cv.VideoCapture(cap_device)
	cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
	cap.set(cv.CAP_PROP_FPS, 60)

	# モデルロード #############################################################
	mp_hands = mp.solutions.hands
	mp_pose = mp.solutions.pose
	hands = mp_hands.Hands(
		static_image_mode = False,
		max_num_hands = 2,
		min_detection_confidence = 0.4,
		min_tracking_confidence = 0.5
	)
	pose = mp_pose.Pose(
		static_image_mode = False,

	)

	# ウィンドウに配置するコンポーネント
	gesture_name = ['ストレッチ', 'ダイヤル', '縦スライド', '横スライド']
	
	col1 = [[sg.Text('ジェスチャ：', justification='center', font=radio_font),
			sg.Text(gesture_name[0], key='gesture_name', justification='center', font=radio_font),
			sg.Text(' 入力範囲：0~', justification='center', font=radio_font),
			sg.Text(input_range, key='-input range-', justification='center', font=radio_font)]]
	col2 = [[sg.Text('', text_color=displaynext_color_code, key='complete', justification='center', font=text_size[1])]]
	col3 = [[sg.Text('', key='-count down-', justification='center', font=radio_font)]]
	col4 = [[sg.Slider(range=(0,input_range), font=radio_font, default_value=random.randrange(0,input_range+1), 
			resolution=1.0, orientation='h', size=(int(screen_w/40), int(screen_h/20)), 
			trough_color=embed_color_code, text_color=default_color_code, key='target_v')]]
	col5 = [[sg.Slider(range=(0,input_range), font=radio_font, default_value=random.randrange(0,input_range+1), 
			resolution=1.0, orientation='h', size=(int(screen_w/40), int(screen_h/20)), 
			trough_color=default_color_code, text_color=default_color_code, key='current_v')]]
	col6 = [[sg.Image(filename='', key='image', size=image_size, visible=visible_flag)]]
	layout = [[sg.Column(i, justification='c')]for i in [col1, col2, col3, col4, col5, col6]]

	# ウィンドウの生成
	window = sg.Window('テスト', layout, return_keyboard_events=True, size=(int(screen_w), int(screen_h)), text_justification='center', resizable=True).Finalize()

	while True:
		# カメラキャプチャ #####################################################
		ret, image = cap.read()
		if not ret:
			break
		image = cv.flip(image, 1) #ミラー表示
		debug_image = copy.deepcopy(image)
		# 検出実施 #############################################################
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		image.flags.writeable = False
		results = hands.process(image)
		image.flags.writeable = True
		if gesture_list[0]==0 and time.time()-pose_interval>4:
			image.flags.writeable = False
			pose_results = pose.process(image)
			image.flags.writeable = True
			if pose_results.pose_landmarks is not None:
				sholder_length = abs(pose_results.pose_landmarks.landmark[11].x-pose_results.pose_landmarks.landmark[12].x)*cap_width*0.7
				pose_cor[0] = (int(pose_results.pose_landmarks.landmark[11].x*cap_width), int(pose_results.pose_landmarks.landmark[11].y*cap_height))
				pose_cor[1] = (int(pose_results.pose_landmarks.landmark[12].x*cap_width), int(pose_results.pose_landmarks.landmark[12].y*cap_height))
				pose_interval = time.time()

		# Hands ###############################################################
		right_hand_landmarks = left_hand_landmarks = None
		if results.multi_hand_landmarks is not None:
			for i, landmarks in enumerate(results.multi_hand_landmarks):
				if results.multi_handedness[i].classification[0].label == "Left":
					right_hand_landmarks = landmarks
				elif results.multi_handedness[i].classification[0].label == "Right":
					left_hand_landmarks = landmarks

		# visible feedback ##########################################################################
		if visible_flag:
			# 実は右手
			if left_hand_landmarks is not None:
				# 手の平重心計算
				cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
				# 外接矩形の計算
				brect = calc_bounding_rect(debug_image, left_hand_landmarks)
				# 描画
				debug_image = draw_hands_landmarks(debug_image, cx, cy, left_hand_landmarks, 'R')
				debug_image = draw_bounding_rect(use_brect, debug_image, brect)
				# 追加
				#手のスケールと描写
				landmark_array = standardize(debug_image, left_hand_landmarks)
				debug_image = draw_righthand(debug_image, landmark_array)
			# # 実は左手
			if right_hand_landmarks is not None:
				# 手の平重心計算
				cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
				# 外接矩形の計算
				brect = calc_bounding_rect(debug_image, right_hand_landmarks)
				# 描画
				debug_image = draw_hands_landmarks(debug_image, cx, cy, right_hand_landmarks, 'L')
				debug_image = draw_bounding_rect(use_brect, debug_image, brect)
				# 追加
				#手のスケールと描写
				landmark_array = standardize(debug_image, right_hand_landmarks)
				debug_image = draw_lefthand(debug_image, landmark_array)

			debug_image = cv.resize(debug_image, dsize=image_size)
			# monitor gesture recognition ####################################
			# confirm 
			rh = "True" if (left_hand_landmarks is not None) and detectHandpose(standardize(image, left_hand_landmarks)) else "False"
			lh = "True" if (right_hand_landmarks is not None) and detectHandpose(standardize(image, right_hand_landmarks)) else "False"
			window['image'].update(data=cv.imencode('.png', debug_image)[1].tobytes())

		# control event #############################################
		event, values = window.read(timeout=0, timeout_key='-timeout-')
		if event == sg.WIN_CLOSED:
			break
		elif event.startswith('Escape'):
			break
		elif event == 'a':
			gesture_flag = 0
			window['gesture_name'].update(gesture_name[gesture_flag])
		elif event == 's':
			gesture_flag = 1
			window['gesture_name'].update(gesture_name[gesture_flag])
		elif event == 'd':
			gesture_flag = 2
			window['gesture_name'].update(gesture_name[gesture_flag])
		elif event == 'f':
			gesture_flag = 3
			window['gesture_name'].update(gesture_name[gesture_flag])
		elif event == 'v':
			visible_flag = False if visible_flag else True
			window['image'].update(visible=visible_flag)

		#ジェスチャ認識#########################################
		if gesture_flag==0:
			gesture_list, post_count = twohand_distance_control(debug_image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count)
		elif gesture_flag==1:
			gesture_list, post_count = dial_control(debug_image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count)
		elif gesture_flag==2:
			gesture_list, post_count = vertical_slide_control(debug_image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count)
		elif gesture_flag==3:
			gesture_list, post_count = horizontal_slide_control(debug_image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count)
		if gesture_list[3]==1:
			if not tmp_flag:
				tmp_flag = True
				tmp_v[0] = tmp_v[2] = int(values['current_v'])
				tmp_v[1] = 0
			else:
				tmp_v[1] = int(format((gesture_list[2]-gesture_list[1])/(sholder_length/input_range), '.0f')) if gesture_flag!=1 else int(format((gesture_list[2]-gesture_list[1])/(max_angle/input_range), '.0f'))
				tmp_v[2] = input_range if tmp_v[0]+tmp_v[1]>input_range else 0 if tmp_v[0]+tmp_v[1]<0 else tmp_v[0]+tmp_v[1]
				
				window['current_v'].update(tmp_v[2])
		else:
			tmp_flag = False
			tmp_v[0] = tmp_v[1] = tmp_v[2] = 0
			gesture_list[0] = gesture_list[1] = gesture_list[2] = gesture_list[3] = 0

		# if not test_flag:
		error = int(values['target_v'])-int(values['current_v'])
		# within acceptable range
		if accept_error>=abs(error):
			window['complete'].update('OK', text_color=color_code)
			if gesture_list[3]==0:
				window['complete'].update('', text_color=default_color_code)
				window["target_v"].update(random.randrange(0, 1+input_range))
				window["current_v"].update(random.randrange(0, 1+input_range))
				gesture_list[3] = 0
		else:
			window['complete'].update('', text_color=default_color_code)

	
	# カメラのリリースなどの後処理
	cap.release()
	cv.destroyAllWindows()
	window.close()


# スライド入力(垂直)
#下方向が数値増加、上方向が数値現象であることに注意
def vertical_slide_control(image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count):
	image_height, image_width, _ = image.shape
	if gesture_list[3]==0:
		post_count = 0
		if (left_hand_landmarks is not None) and detectHandpose(standardize(image, left_hand_landmarks)):
			gesture_list[0] = 1
			gesture_list[2] = gesture_list[1] = min(int(left_hand_landmarks.landmark[9].y*image_height), image_width-1)
			# gesture_list[1] = 0
			gesture_list[3] = 1
		elif (right_hand_landmarks is not None) and detectHandpose(standardize(image, right_hand_landmarks)):
			gesture_list[0] = 2
			gesture_list[2] = gesture_list[1] = min(int(right_hand_landmarks.landmark[9].y*image_height), image_width-1)
			# gesture_list[1] = 0
			gesture_list[3] = 1
		else:
			gesture_list[1] = gesture_list[2] = gesture_list[3] = 0

	else:
		land = left_hand_landmarks if gesture_list[0]==1 else right_hand_landmarks
		if (land is not None) and detectHandpose(standardize(image, land)):
			post_count = 0
			gesture_list[1] = min(int(land.landmark[9].y*image_height), image_width-1)
		else:
			post_count = post_count + 1
			if post_count>=postponement:
				gesture_list[0] = gesture_list[3] = post_count = 0
		# else:
		# 	gesture_list[0] = gesture_list[3] = post_count = 0
	return gesture_list, post_count


#スライド入力(水平)
def horizontal_slide_control(image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count):
	image_height, image_width, _ = image.shape
	if gesture_list[3]==0:
		post_count = 0
		if (left_hand_landmarks is not None) and detectHandpose(standardize(image, left_hand_landmarks)):
			gesture_list[0] = 1
			gesture_list[1] = gesture_list[2] = min(int(left_hand_landmarks.landmark[9].x*image_width), image_width-1)
			gesture_list[3] = 1
		elif (right_hand_landmarks is not None) and detectHandpose(standardize(image, right_hand_landmarks)):
			gesture_list[0] = 2
			gesture_list[1] = gesture_list[2] = min(int(right_hand_landmarks.landmark[9].x*image_width), image_width-1)
			gesture_list[3] = 1
	else:
		land = left_hand_landmarks if gesture_list[0]==1 else right_hand_landmarks
		if (land is not None) and detectHandpose(standardize(image, land)):
			post_count = 0
			gesture_list[2] =min(int(land.landmark[9].x*image_width), image_width-1)
		else:
			post_count = post_count + 1
			if post_count>=postponement:
				gesture_list[0] = gesture_list[3] = post_count = 0
	return gesture_list, post_count

#両手の距離による入力手法
#カメラからの距離によるスケーリングが必要かもしれない
def twohand_distance_control(image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count):
	if (right_hand_landmarks is not None) and (left_hand_landmarks is not None) and detectHandpose(standardize(image, left_hand_landmarks)) and  detectHandpose(standardize(image, right_hand_landmarks)):
		image_height, image_width, _ = image.shape
		post_count = 0
		tmp_flag = True if gesture_list[0]==0 else False
		# gesture_list[3] = 1 if gesture_list[0]==1 else 0
		gesture_list[0] = gesture_list[3] = 1
		delta_x = min(int(left_hand_landmarks.landmark[2].x*image_width), image_width-1)-min(int(right_hand_landmarks.landmark[2].x*image_width), image_width-1)
		delta_y = min(int(left_hand_landmarks.landmark[2].y*image_height), image_height-1)-min(int(right_hand_landmarks.landmark[2].y*image_height), image_height-1)
		# if abs(delta_x)<=max_dis*1.1:
		if tmp_flag:
			gesture_list[1] = gesture_list[2] = np.linalg.norm([delta_x, delta_y])
		else:
			gesture_list[2] = np.linalg.norm([delta_x, delta_y])
	elif gesture_list[3]==1:
		post_count = post_count+1
		if post_count>=postponement:
			post_count = gesture_list[0] = gesture_list[3] = 0
	else:
		post_count = gesture_list[0] = gesture_list[3] = 0
	# print(gesture_list)
	return gesture_list, post_count

#ダイアルジェスチャ(回転)による入力手法
def dial_control(image, right_hand_landmarks, left_hand_landmarks, gesture_list, postponement, post_count):
	if gesture_list[0]==0:
		if (left_hand_landmarks is not None) and detectHandpose(standardize(image, left_hand_landmarks)):
			gesture_list[0] = 1
			gesture_list[1] = gesture_list[2] = calc_hand_gesture_list(image, left_hand_landmarks)
			gesture_list[3] = 1
		elif (right_hand_landmarks is not None) and detectHandpose(standardize(image, right_hand_landmarks)):
			gesture_list[0] = 2
			gesture_list[1] = gesture_list[2] = calc_hand_gesture_list(image, right_hand_landmarks)
			gesture_list[3] = 1
	elif gesture_list[3]==1:
		land = left_hand_landmarks if gesture_list[0]==1 else right_hand_landmarks
		if (land is not None) and detectHandpose(standardize(image, land)):
			tmp = calc_hand_gesture_list(image, land)
			gesture_list[2] = tmp
		else:
			post_count = post_count+1
			if post_count>=postponement:
				gesture_list[0] = gesture_list[3] = post_count = 0
	else:
		gesture_list[0] = gesture_list[3] = post_count = 0
	return gesture_list, post_count

def calc_hand_gesture_list(image, landmarks):
	lines = []
	image_height, image_width, _ = image.shape
	for index, landmark in enumerate(landmarks.landmark):
		if index==0 or index==9:
			landmark_x = min(int(landmark.x * image_width), image_width - 1)
			landmark_y = min(int(landmark.y * image_height), image_height - 1)
			lines.append((landmark_x, landmark_y))
	return calc_gesture_list(lines)

#OKポーズの数値確認用
def checkOK(landmark_array):
	fingers = fingerstate(landmark_array)
	vec_dis = [landmark_array[4]-landmark_array[8], landmark_array[4]-landmark_array[12], landmark_array[4]-landmark_array[16], landmark_array[4]-landmark_array[20]]
	dis = [np.linalg.norm([vec_dis[0][0], vec_dis[0][1]]), np.linalg.norm([vec_dis[1][0], vec_dis[1][1]]), np.linalg.norm([vec_dis[2][0], vec_dis[2][1]]), np.linalg.norm([vec_dis[3][0], vec_dis[3][1]])]
	return dis

#2本のベクトルのなす角を求める
def gesture_list_2fingers(base, a, b):
	vec_a = [a[0]-base[0], a[1]-base[1]]
	vec_b = [b[0]-base[0], b[1]-base[1]]
	# deg = np.degrees(np.arccos((vec_a[0]*vec_b[0]+vec_a[1]*vec_b[1])/(np.linalg.norm([vec_a[0], vec_a[1]])*np.linalg.norm([vec_b[0], vec_b[1]]))))
	return np.degrees(np.arccos((vec_a[0]*vec_b[0]+vec_a[1]*vec_b[1])/(np.linalg.norm([vec_a[0], vec_a[1]])*np.linalg.norm([vec_b[0], vec_b[1]]))))

#指の曲がりを計算
def fingerstate(landmark_array):
	fingers = []
	fingers.append(1) if gesture_list_2fingers(landmark_array[2], landmark_array[4], landmark_array[5])<50 else fingers.append(0)
	fingers.append(1) if landmark_array[6][1] > landmark_array[8][1] else fingers.append(0)
	fingers.append(1) if landmark_array[10][1] > landmark_array[12][1] else fingers.append(0)
	fingers.append(1) if landmark_array[14][1] > landmark_array[16][1] else fingers.append(0)
	fingers.append(1) if landmark_array[18][1] > landmark_array[20][1] else fingers.append(0)
	return fingers

#ジェスチャ入力可能なポーズかを判定する
def detectHandpose(landmark_array):
	thumb_first = landmark_array[4]-landmark_array[8]
	if np.linalg.norm([thumb_first[0], thumb_first[1]])<0.16 and landmark_array[11][1]<landmark_array[9][1] and landmark_array[15][0]<landmark_array[13][1] and landmark_array[19][1]<landmark_array[17][1]:
		return True
	else:
		return False

def distance_cm(image_width, image_height, land1, land2):
	delta_x = min(int(land1.x*image_width), image_width-1)-min(int(land2.x*image_width), image_width-1)
	delta_y = min(int(land1.y*image_height), image_height-1)-min(int(land2.y*image_height), image_height-1)
	return np.linalg.norm([delta_x, delta_y])

#line_c -> ((x1,y1), (x2,y2))
#∠AOBの計算
def calc_gesture_list(line_c):
	x, y = line_c[1][0]-line_c[0][0], line_c[1][1]-line_c[0][1]
	return np.degrees(np.arctan2(line_c[1][0]-line_c[0][0],line_c[0][1]-line_c[1][1]))

#landmarkの座標を0~1にスケーリング
def scaling(landmark_array):
	#座標内の最大値最小値を取得
	cmax, cmin = np.amax(landmark_array, axis=0), np.amin(landmark_array, axis=0)
	landmark_new = np.empty(tuple(landmark_array.shape))
	length = cmax[0]-cmin[0] if (cmax[0]-cmin[0]>cmax[1]-cmin[1]) else cmax[1]-cmin[1]

	#スケーリング
	for i in range(landmark_array.shape[0]):
		landmark_new[i][0] = (landmark_array[i][0]-cmin[0])/length
		landmark_new[i][1] = (landmark_array[i][1]-cmin[1])/length
	return landmark_new

#手の幅と高さ取得
def calc_WH(image, landmark):
	landmark_array = hand2array(image, landmark)
	cmax, cmin = np.amax(landmark_array, axis=0), np.amin(landmark_array, axis=0)
	width, height = cmax[0]-cmin[0], cmax[1]-cmin[1]
	return  width, height #幅、高さ

#landmarkからlandmark_arrayに変換
def hand2array(image, landmark):
	image_height, image_width, _ = image.shape
	landmark_array = np.empty((0,2), int)
	gesture_list_array = np.empty((0,1), float)
	#回転部
	for index, l in enumerate(landmark.landmark):
		landmark_x = min(int(l.x * image_width), image_width - 1)
		landmark_y = min(int(l.y * image_height), image_height - 1)
		if index == 0:
			base_l = (landmark_x, landmark_y)
			gesture_list_array = np.append(gesture_list_array, 0)
		else:
			gesture_list_array = np.append(gesture_list_array, np.arctan2(landmark_x-base_l[0], landmark_y-base_l[1]))
	base_gesture_list = copy.copy(gesture_list_array[9])
	for index, l in enumerate(landmark.landmark):
		landmark_x = min(int(l.x * image_width), image_width - 1)
		landmark_y = min(int(l.y * image_height), image_height - 1)
		gesture_list_array[index] = gesture_list_array[index] - base_gesture_list
		length = np.linalg.norm([landmark_x-base_l[0], landmark_y-base_l[1]])
		if length != 0:
			coors = np.array([int(length*np.sin(gesture_list_array[index])), int(length*np.cos(gesture_list_array[index]))])
		else:
			coors = np.array([0, 0])
		landmark_array = np.append(landmark_array, np.array([coors]), axis=0)
	return landmark_array

#手の座標標準化
def standardize(image, landmark):
	image_height, image_width, _ = image.shape
	landmark_array = np.empty((0,2), int)
	gesture_list_array = np.empty((0,1), float)
	#回転部
	for index, l in enumerate(landmark.landmark):
		landmark_x = min(int(l.x * image_width), image_width - 1)
		landmark_y = min(int(l.y * image_height), image_height - 1)
		if index == 0:
			base_l = (landmark_x, landmark_y)
			gesture_list_array = np.append(gesture_list_array, 0)
		else:
			gesture_list_array = np.append(gesture_list_array, np.arctan2(landmark_x-base_l[0], landmark_y-base_l[1]))
	base_gesture_list = copy.copy(gesture_list_array[9])
	for index, l in enumerate(landmark.landmark):
		landmark_x = min(int(l.x * image_width), image_width - 1)
		landmark_y = min(int(l.y * image_height), image_height - 1)
		gesture_list_array[index] = gesture_list_array[index] - base_gesture_list
		length = np.linalg.norm([landmark_x-base_l[0], landmark_y-base_l[1]])
		if length != 0:
			coors = np.array([int(length*np.sin(gesture_list_array[index])), int(length*np.cos(gesture_list_array[index]))])
		else:
			coors = np.array([0, 0])
		landmark_array = np.append(landmark_array, np.array([coors]), axis=0)
	#スケーリングして返す
	return scaling(landmark_array)

#ワイプに右手を描写
def draw_righthand(image, landmark_array):
	(image_height, image_width, _) = image.shape
	wipe_size = 100
	cv.rectangle(image, (image_width-wipe_size, image_height-wipe_size), (image_width, image_height), (0,0,0), -1)
	#ライン描写
	for a,b in [(0,1),(1,2),(2,3),(3,4),(0,17),(17,13),(13,9),(9,5),(5,2),(17,18),(18,19),(19,20),(13,14),(14,15),(15,16),(9,10),(10,11),(11,12),(5,6),(6,7),(7,8)]:
		cv.line(image, (int((image_width - wipe_size) + wipe_size*landmark_array[a][0]),
		int(image_height-(landmark_array[a][1]*wipe_size))),
				(int((image_width - wipe_size) + wipe_size*landmark_array[b][0]),
				int(image_height-(landmark_array[b][1]*wipe_size))), (0, 255, 0), 1)
	return image

#ワイプに左手を描写
def draw_lefthand(image, landmark_array):
	(image_height, image_width, _) = image.shape
	wipe_size = 100
	cv.rectangle(image, (0, image_height-wipe_size), (wipe_size, image_height), (0,0,0), -1)
	#ライン描写
	for a,b in [(0,1),(1,2),(2,3),(3,4),(0,17),(17,13),(13,9),(9,5),(5,2),(17,18),(18,19),(19,20),(13,14),(14,15),(15,16),(9,10),(10,11),(11,12),(5,6),(6,7),(7,8)]:
		cv.line(image, (int(wipe_size*landmark_array[a][0]), int(image_height-(landmark_array[a][1]*wipe_size))),
				(int(wipe_size*landmark_array[b][0]), int(image_height-(landmark_array[b][1]*wipe_size))), (0, 255, 0), 1)
	return image

def calc_palm_moment(image, landmarks):
	image_width, image_height = image.shape[1], image.shape[0]

	palm_array = np.empty((0, 2), int)

	for index, landmark in enumerate(landmarks.landmark):
		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)

		landmark_point = [np.array((landmark_x, landmark_y))]

		if index == 0:  # 手首1
			palm_array = np.append(palm_array, landmark_point, axis=0)
		if index == 1:  # 手首2
			palm_array = np.append(palm_array, landmark_point, axis=0)
		if index == 5:  # 人差指：付け根
			palm_array = np.append(palm_array, landmark_point, axis=0)
		if index == 9:  # 中指：付け根
			palm_array = np.append(palm_array, landmark_point, axis=0)
		if index == 13:  # 薬指：付け根
			palm_array = np.append(palm_array, landmark_point, axis=0)
		if index == 17:  # 小指：付け根
			palm_array = np.append(palm_array, landmark_point, axis=0)
	M = cv.moments(palm_array)
	cx, cy = 0, 0
	if M['m00'] != 0:
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])

	return cx, cy


def calc_bounding_rect(image, landmarks):
	image_width, image_height = image.shape[1], image.shape[0]

	landmark_array = np.empty((0, 2), int)

	for _, landmark in enumerate(landmarks.landmark):
		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)

		landmark_point = [np.array((landmark_x, landmark_y))]

		landmark_array = np.append(landmark_array, landmark_point, axis=0)

	x, y, w, h = cv.boundingRect(landmark_array)

	return [x, y, x + w, y + h]


def draw_hands_landmarks(image, cx, cy, landmarks, handedness_str='R'):
	image_width, image_height = image.shape[1], image.shape[0]

	landmark_point = []

	# キーポイント
	for index, landmark in enumerate(landmarks.landmark):
		if landmark.visibility < 0 or landmark.presence < 0:
			continue

		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)
		landmark_z = landmark.z

		landmark_point.append((landmark_x, landmark_y))

		if index == 0:  # 手首1
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
			# cv.putText("{}".format(landmark_z), )
		if index == 1:  # 手首2
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 2:  # 親指：付け根
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 3:  # 親指：第1関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 4:  # 親指：指先
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
			cv.circle(image, (landmark_x, landmark_y), 12, (188, 5, 245), 2)
		if index == 5:  # 人差指：付け根
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 6:  # 人差指：第2関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 7:  # 人差指：第1関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 8:  # 人差指：指先
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
			cv.circle(image, (landmark_x, landmark_y), 12, (188, 5, 245), 2)
		if index == 9:  # 中指：付け根
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 10:  # 中指：第2関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 11:  # 中指：第1関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 12:  # 中指：指先
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
			cv.circle(image, (landmark_x, landmark_y), 12, (188, 5, 245), 2)
		if index == 13:  # 薬指：付け根
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 14:  # 薬指：第2関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 15:  # 薬指：第1関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 16:  # 薬指：指先
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
			cv.circle(image, (landmark_x, landmark_y), 12, (188, 5, 245), 2)
		if index == 17:  # 小指：付け根
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 18:  # 小指：第2関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 19:  # 小指：第1関節
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 20:  # 小指：指先
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
			cv.circle(image, (landmark_x, landmark_y), 12, (188, 5, 245), 2)

	# 接続線
	if len(landmark_point) > 0:
		# 親指
		cv.line(image, landmark_point[2], landmark_point[3], (188, 5, 245), 2)
		cv.line(image, landmark_point[3], landmark_point[4], (188, 5, 245), 2)

		# 人差指
		cv.line(image, landmark_point[5], landmark_point[6], (188, 5, 245), 2)
		cv.line(image, landmark_point[6], landmark_point[7], (188, 5, 245), 2)
		cv.line(image, landmark_point[7], landmark_point[8], (188, 5, 245), 2)

		# 中指
		cv.line(image, landmark_point[9], landmark_point[10], (188, 5, 245), 2)
		cv.line(image, landmark_point[10], landmark_point[11], (188, 5, 245), 2)
		cv.line(image, landmark_point[11], landmark_point[12], (188, 5, 245), 2)

		# 薬指
		cv.line(image, landmark_point[13], landmark_point[14], (188, 5, 245), 2)
		cv.line(image, landmark_point[14], landmark_point[15], (188, 5, 245), 2)
		cv.line(image, landmark_point[15], landmark_point[16], (188, 5, 245), 2)

		# 小指
		cv.line(image, landmark_point[17], landmark_point[18], (188, 5, 245), 2)
		cv.line(image, landmark_point[18], landmark_point[19], (188, 5, 245), 2)
		cv.line(image, landmark_point[19], landmark_point[20], (188, 5, 245), 2)
 
		# 手の平
		cv.line(image, landmark_point[0], landmark_point[1], (188, 5, 245), 2)
		cv.line(image, landmark_point[1], landmark_point[2], (188, 5, 245), 2)
		cv.line(image, landmark_point[2], landmark_point[5], (188, 5, 245), 2)
		cv.line(image, landmark_point[5], landmark_point[9], (188, 5, 245), 2)
		cv.line(image, landmark_point[9], landmark_point[13], (188, 5, 245), 2)
		cv.line(image, landmark_point[13], landmark_point[17], (188, 5, 245), 2)
		cv.line(image, landmark_point[17], landmark_point[0], (188, 5, 245), 2)

	# 重心 + 左右
	if len(landmark_point) > 0:
		cv.circle(image, (cx, cy), 12, (80, 44, 110), 2)
		cv.putText(image, handedness_str, (cx - 6, cy + 6),
					cv.FONT_HERSHEY_SIMPLEX, 0.6, (80, 44, 110), 2, cv.LINE_AA)

	return image

def draw_bounding_rect(use_brect, image, brect):
	if use_brect:
		# 外接矩形
		cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
						(188, 5, 245), 2)

	return image

if __name__ == '__main__':
	main()