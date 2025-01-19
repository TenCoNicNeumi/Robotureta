import cv2
import mediapipe as mp
import numpy as np
import matplotlib   as plt
import threading
import time
import torch
import random

# Načtení předtrénovaného modelu YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start (tic)
start_time = time.time()
# Inicializace MediaPipe pro detekci póz
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# pose = mp_pose.Pose()
pose = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.8)


# Inicializace nástroje pro vykreslování klíčových bodů
mp_drawing = mp.solutions.drawing_utils

# Otevření webkamery
cap = cv2.VideoCapture(0)

x1, x2, y1, y2 = (0,0,0,0)

saved_color = None

# Definuj ID bodů, které chceš sledovat (např. ramena, lokty, kolena)
body_points_to_track = [0]  
deact = False
tracked_person_id = None
width = 1280  # Šířka
height = 720  # Výška
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
reset_target = False
roi = None

while True:
    ret, frame = cap.read()  # Capture the frame first
    if not ret:
        print("Nelze získat snímek z webkamery.")
        break
    results = model(frame)

    # Získání detekcí (class 'person' má ID 0)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    # Filtrování pouze osob (class == 0)
    people = [d for d in detections if int(d[5]) == 0]
    if len(people) > 0:
        # Pokud jsou detekováni lidé
        selected_person = random.choice(people)
        osoba_x1, osoba_y1, osoba_x2, osoba_y2 = map(int, selected_person[:4])
        osoba_zona = osoba_x1, osoba_y1, osoba_x2, osoba_y2
    else:
        # Pokud nejsou detekováni žádní lidé, nastavte výchozí hodnoty
        osoba_x1, osoba_y1, osoba_x2, osoba_y2 = None, None, None, None
        osoba_zona = None
        
    

    while cap.isOpened() and osoba_zona is not None:
        ret, frame = cap.read()
        if not ret:
            print("Nelze získat snímek z webkamery.")
            break
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

        # Filtrování pouze osob (class == 0)
        people = [d for d in detections if int(d[5]) == 0]

        
        if len(people) > 0:
            # Pokud jsou detekováni lidé
            # selected_person = random.choice(people)
            osoba_x1, osoba_y1, osoba_x2, osoba_y2 = map(int, selected_person[:4])
            osoba_zona = osoba_x1, osoba_y1, osoba_x2, osoba_y2
        else:
            # Pokud nejsou detekováni žádní lidé, nastavte výchozí hodnoty
            osoba_x1, osoba_y1, osoba_x2, osoba_y2 = None, None, None, None
        osoba_zona = frame[osoba_y1:osoba_y2, osoba_x1:osoba_x2]
        rgb_frame = cv2.cvtColor(osoba_zona, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if osoba_x1 is not None and osoba_y1 is not None and osoba_x2 is not None and osoba_y2 is not None:
            osoba_zona = frame[osoba_y1:osoba_y2, osoba_x1:osoba_x2]
            rgb_frame = cv2.cvtColor(osoba_zona, cv2.COLOR_BGR2RGB)

            result = pose.process(rgb_frame)
            
        else:
            print("Invalid osoba_zona coordinates, skipping frame.")
            continue
        if len(people) <= 0:
            # Pokud nejsou detekováni žádní lidé
            osoba_x1, osoba_y1, osoba_x2, osoba_y2 = None, None, None, None
            
        # Převod snímku na RGB (MediaPipe pracuje s RGB)
        rgb_frame = cv2.cvtColor(osoba_zona, cv2.COLOR_BGR2RGB)                 # zmenit na osoba osoba_zona
        # Detekce póz na aktuálním snímku
        result = pose.process(rgb_frame)
        
        if result.pose_landmarks is None:
            cv2.imshow('MediaPipe Pose Tracking', rgb_frame)
            continue
        
        



        if result.pose_landmarks:
    #--------------------------------------------------------------------------------------------------
            all_landmarks = result.pose_landmarks.landmark
            

            
            # Inicializace proměnných
            min_distance = float('inf')
            closest_person_id = None
            max_distance = float('-inf')
            farthest_person_id = None

            # Projděte všechny landmarky (detekované osoby)
            for person_id, landmark in enumerate(all_landmarks):
                if person_id == tracked_person_id:  # Přeskočte aktuálně sledovanou osobu
                    continue

                # Vypočítejte vzdálenost od kamery (z-souřadnice)
                distance = landmark.z
                
                if distance > max_distance:  # Najdi osobu nejdál od kamery
                    max_distance = distance
                    farthest_person_id = person_id
                if distance < min_distance:
                    min_distance = distance
                    closest_person_id = person_id
                
                if reset_target:
                    all_landmarks = result.pose_landmarks.landmark
                    
                    # Inicializace hledání nejbližší osoby
                    min_distance = float('inf')
                    closest_person_id = None

                    # Projdeme všechny landmarky
                    for person_id, landmark in enumerate(all_landmarks):
                        distance = landmark.z  # Z-souřadnice označuje vzdálenost
                        if distance < min_distance:  # Najít nejbližší osobu
                            min_distance = distance
                            closest_person_id = person_id

                    # Pokud jsme našli novou osobu, začni ji sledovat
                    if closest_person_id is not None:
                        tracked_person_id = closest_person_id
                        print(f"Sledování přepnuto na novou osobu: ID {tracked_person_id}")

                    # Resetovat požadavek na hledání
                    reset_target = False

                    # Vykresli sledovanou osobu
                    if tracked_person_id is not None:
                        tracked_person_landmark = result.pose_landmarks.landmark[tracked_person_id]
                        
                        #cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # Modrý kruh na sledované osobě
                        #cv2.putText(frame, f"Sledovana osoba: ID {tracked_person_id}",
                        #            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            


            
            # Pokud najdete jinou osobu, přepněte na ni
            if closest_person_id is not None:
                tracked_person_id = closest_person_id

            # Vykreslení aktuálně sledované osoby
            if tracked_person_id is not None:
                tracked_person_landmark = result.pose_landmarks.landmark[tracked_person_id]
                x, y = int(tracked_person_landmark.x * rgb_frame.shape[1]+osoba_x1), int(tracked_person_landmark.y * rgb_frame.shape[0]+osoba_y1)
                
                
    #---------------------------------------------------------------------------------------------------------------------------------
            # Získání bodu pravého ramene (bod 12)
            right_shoulder = result.pose_landmarks.landmark[12]
            x12, y12 = int(right_shoulder.x * rgb_frame.shape[1]+osoba_x1), int(right_shoulder.y * rgb_frame.shape[0]+osoba_y1)

            # Definuj oblast pro výpočet převládající barvy (např. čtverec o rozměru 50x50 pixelů kolem ramene)
            roi_size = 70
            x1 = max(0, x12 + 0)   #- roi_size // 2
            y1 = max(0, y12 + 0)
            x2 = min(frame.shape[1], x12 + roi_size)
            y2 = min(frame.shape[0], y12 + roi_size)

            # Kontrola, zda ROI má platnou velikost
            if x1 < x2 and y1 < y2:
                roi = frame[y1:y2, x1:x2]
            
                        
                # Kontrola, zda ROI není prázdná
                if roi.size > 0:
                    # Výpočet histogramu pro každý barevný kanál v ROI
                    hist_b = cv2.calcHist([roi], [0], None, [256], [0, 256])
                    hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
                    hist_r = cv2.calcHist([roi], [2], None, [256], [0, 256])

                    # Získání nejčastější barvy
                    predominant_blue = int(np.argmax(hist_b))
                    predominant_green = int(np.argmax(hist_g))
                    predominant_red = int(np.argmax(hist_r))
                        
            else:
                roi = None  # Pokud je ROI mimo obraz, nastavíme ji na None
                predominant_blue = None
                predominant_green = None
                predominant_red = None

            # Zobrazení převládající barvy v oblasti
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'BGR: ({predominant_blue}, {predominant_green}, {predominant_red})', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            

        # Kontrola stisknutí tlačítkaqqqqqqq
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Stisknutí 's' pro uložení barvy
            saved_color = (predominant_blue, predominant_green, predominant_red)


        
        



        

        if result.pose_landmarks is not None:
            right_shoulder = result.pose_landmarks.landmark[12]
            x12, y12 = int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])
            
            all_people_landmarks = []

            
            min_distance = float('inf')  
            closest_person_landmarks = None  

            
            for id, landmark in enumerate(result.pose_landmarks.landmark):
                h, w, _ = rgb_frame.shape
                x, y, z = int(landmark.x * w), int(landmark.y * h), landmark.z

                if z < min_distance:
                    min_distance = z
                    closest_person_landmarks = result.pose_landmarks

            
            if closest_person_landmarks:
                # mp_drawing.draw_landmarks(frame, closest_person_landmarks, mp_pose.POSE_CONNECTIONS)

                
                for id, landmark in enumerate(closest_person_landmarks.landmark):

                    if result.pose_landmarks:
                        # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        if result.pose_landmarks:
                            # Přepočet landmarků z osoba_zona na původní frame
                            for landmark in result.pose_landmarks.landmark:
                                # Přepočet souřadnic
                                x = int(landmark.x * osoba_zona.shape[1]) + osoba_x1
                                y = int(landmark.y * osoba_zona.shape[0]) + osoba_y1

                                # Vykreslení přepočteného bodu na původním framu
                                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)


                        if id in body_points_to_track:
                                h, w, d = frame.shape
                                
                                x1, y1 = int(w/2 - 20) , int(h/2 - 20)  # Levý horní roh
                                x2, y2 = int(w/2 + 20) , int(h/2 + 20) # Pravý dolní roh
                                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2) 
                                BodNos = result.pose_landmarks.landmark[0] 
                                BodLZ = result.pose_landmarks.landmark[15] 
                                BodPZ = result.pose_landmarks.landmark[16] 
                                x_nos, y_nos = int(BodNos.x * rgb_frame.shape[1]+osoba_x1), int(BodNos.y * rgb_frame.shape[0]+osoba_y1)
                                x_lz, y_lz = int(BodLZ.x * rgb_frame.shape[1] ), int(BodLZ.y * frame.shape[0])
                                x_pz, y_pz = int(BodPZ.x * rgb_frame.shape[1]), int(BodPZ.y * frame.shape[0])
                                #x, y = int(BodNos.x * w), int(BodNos.y * h)


                                if roi is not None:
                                    if roi.size > 0:
                                        # Výpočet průměrné barvy v ROI
                                        average_color = roi.mean(axis=0).mean(axis=0)  # Průměrná barva (BGR)
                                        predominant_color = tuple(map(int, average_color))  # Převod na celá čísla

                                        # Zobrazení převládající barvy
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(frame, f'BGR: ({predominant_color[0]}, {predominant_color[1]}, {predominant_color[2]})', 
                                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                        # Porovnání s uloženou barvou
                                        if saved_color:
                                            threshold = 50  # Maximální povolený rozdíl mezi barvami

                                            # Funkce pro porovnání barev
                                            def is_similar_color(color1, color2, threshold):
                                                return all(abs(c1 - c2) <= threshold for c1, c2 in zip(color1, color2))

                                            # Porovnání uložené a aktuální barvy
                                            if is_similar_color(saved_color, predominant_color, threshold):
                                                deact = True
                                                print('stop')
                                                cv2.putText(frame, f"Ne",
                                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                                break
                                                
                                        elif (y_nos * h) > (y_lz * h) or (y_nos * h) > (y_pz * h):
                                            print('Stop')
                                            cv2.putText(frame, f"Ne",
                                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                            break


                                        elif x_nos > w/2-20 and x_nos < w/2+20 and y_nos > h/2-20 and y_nos < h/2+20:
                                            print ('Pew')
                                            cv2.putText(frame, f"Ano",
                                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        

                                        cv2.rectangle(frame, (osoba_x1, osoba_y1), (osoba_x2, osoba_y2), (255, 255, 255), 2)


        
        cv2.imshow('MediaPipe Pose Tracking', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print(saved_color)
cap.release()
cv2.destroyAllWindows()

