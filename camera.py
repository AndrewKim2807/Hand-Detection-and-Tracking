import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=8)  # Set max_num_hands to the desired number of hands
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0

while True:
    success, img = cap.read()

    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    num_hands = 0  # Initialize the number of detected hands

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                cv2.circle(img, (cx, cy), 15, (139, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
            
            num_hands += 1  # Increment the number of detected hands

    message = f"{num_hands} hand(s) detected"  # Generate the message based on the number of hands
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, message, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 0, 0), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 0, 0), 2)
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) == 27:  # Check if the "Esc" key is pressed
        break

cap.release()
cv2.destroyAllWindows()
