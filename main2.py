import cv2
import mediapipe as mp

def main():
    finger_tip_indices = [4, 8, 12, 16, 20]
    count = 0

    # Inizializza il modulo MediaPipe per il rilevamento delle mani
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Leggi l'immagine dalla webcam
        success, frame = cap.read()
        if not success:
            continue

        # Converte il frame in scala di grigi
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Esegui il rilevamento della mano
        results = hands.process(frame_rgb)

        # Verifica se sono stati rilevati punti della mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):
                    # Ottieni le coordinate normalizzate (valori tra 0 e 1)
                    x, y, _ = landmark.x, landmark.y, landmark.z

                    # Converti le coordinate normalizzate in pixel
                    h, w, _ = frame.shape
                    x_pixel, y_pixel = int(x * w), int(y * h)

                    # Disegna un cerchio sul punto della mano
                    cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)

                    # Stampa le coordinate nella console
                    if id == 7 or id == 8:
                        print(f"Coordinate del punto: {id}: ({x}, {y})")
                        # Aggiungi un testo alla finestraq
                    for pippo in range(4):
                        if hand_landmarks.landmark[(pippo + 2) * 4].y < hand_landmarks.landmark[(pippo + 2) * 4 - 1].y:
                            count += 1
                    if hand_landmarks.landmark[(1) * 4].x < hand_landmarks.landmark[(1) * 4 - 1].x:
                        count += 1
                    cv2.putText(frame, f"{count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Mostra il frame con le coordinate dei punti della mano e il testo
                    cv2.imshow('Hand Landmarks', frame)
                    count = 0

        # Mostra il frame con le coordinate dei punti della mano
        cv2.imshow('Hand Landmarks', frame)

        # Esci se viene premuto il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
