"""
カメラ番号を識別する関数
"""
import cv2

#カメラのidを確認
for i1 in range(0, 20): 
    cap1 = cv2.VideoCapture( i1, cv2.CAP_DSHOW )
    if cap1.isOpened(): 
        print("VideoCapture(", i1, ") : Found")
        cam_id = i1
    # else:
    #     print("VideoCapture(", i1, ") : None")
    cap1.release() 
    
#フレームサイズ確認
cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

print (cap.get(cv2.CAP_PROP_FPS))
print (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
# cv2.destroy.AllWindows()