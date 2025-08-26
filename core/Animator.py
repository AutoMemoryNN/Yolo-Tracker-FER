import sys
import time
import threading
from typing import Dict, Any

import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QGridLayout,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage
from openai import OpenAI

from core.Interpreter import Interpreter, TopEmotional


class Animator(threading.Thread):
    def __init__(self, interpreter: Interpreter):
        super().__init__(daemon=True)
        self.interpreter = interpreter
        self.chatgpt_client = OpenAI()
        self.latest_data: Dict[str, Any] = {}
        self.commentary: str = ""

    def run(self) -> None:
        last_state = 0.0
        last_chat = 0.0
        while True:
            now = time.monotonic()

            # Actualiza snapshot de estado cada 1s
            if now - last_state >= 1.0:
                data = {
                    "threshold_top": self.interpreter.threshold_top or {},
                    "avg_emotion": self.interpreter.avg_persons_emotion or {},
                    "num_persons": len(self.interpreter.persons_by_id or {}),
                }
                self.latest_data = data
                print("Animator snapshot:", data)
                last_state = now

            # Genera comentario cada 15s
            if now - last_chat >= 15.0:
                try:
                    prompt = self._build_prompt(self.latest_data)
                    print("Prompt para ChatGPT:", prompt)

                    response = self.chatgpt_client.chat.completions.create(
                        model="gpt-5-nano",  # o4-mini
                        messages=[
                            {
                                "role": "system",
                                "content": "Eres un animador de eventos.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                    print("Respuesta de ChatGPT:", response)

                    if response and response.choices:
                        content = response.choices[0].message.content
                        if content:
                            self.commentary = content.strip()
                            print("Comentario generado:", self.commentary)
                        else:
                            self.commentary = "¡Qué evento tan interesante!"
                            print("Contenido vacío, usando comentario por defecto.")
                    else:
                        self.commentary = "¡Qué evento tan interesante!"
                        print("Respuesta vacía, usando comentario por defecto.")
                except Exception as e:
                    print("Error generando comentario:", e)
                finally:
                    last_chat = now

            time.sleep(0.05)

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """
        Construye el contexto que se le manda a ChatGPT
        """
        base_context = (
            "Eres un animador en un evento universitario. "
            "El presentador es Juan Pablo Camargo, líder del semillero de IA. "
            "Estudiantes de grado 11 visitan la universidad Escuela colombiana de Ingenieria Julio Garavito. "
            "Tu rol es comentar en tiempo real de manera entretenida, positiva y cercana.\n\n"
            "Datos actuales:\n"
            "No hagas siempre el mismo comentario, saluda de vez en cuando, y recuerda enfocarte en el publico, menciona cuantas personas se encuentran y cosas por el estilo."
            f"- Personas detectadas (rostros detectados): {data.get('num_persons', 0)}\n"
            f"- Emociones promedio (emociones de los ultimos 5 segundos del publico): {data.get('avg_emotion', {})}\n"
            f"- Top emociones (emociones mas intensas en el momento solo son interesantes si superan el umbral 0.94 o si no hay mas temas): {list(data.get('threshold_top', {}).keys())}\n\n"
            "Si se incluye una imagen de una persona aleatoria, haz un cumplido amable y breve.\n"
            f"Este fue tu comentario anterior: {self.commentary}"
            "Recuerda ser breve, no menciones dos veces seguidas lo mismo."
            "IMPORTANTE, la dinamica es que los estudiantes hagan emociones frente a la camara y vean como la IA la clasifica, si vez caras de susto o enojadas no es malo, es parte de la dinamica y podrias jugar con ello."
            "No tienes que mencionar mi nombre ni la universidad tan seguido, porque los estudiantes ya lo ven."
        )
        return base_context

    def start_windows(self, interpreter: Interpreter) -> None:
        """Convenience helper to start the interpreter thread and show Qt windows."""
        t = threading.Thread(target=interpreter.interpret, daemon=True)
        t.start()

        self.start()

        app = QApplication.instance() or QApplication(sys.argv)

        anim_win = AnimadorWindow(self)
        anim_win.resize(1000, 400)
        anim_win.show()

        top_win = TopEmotionWindow(self)
        top_win.resize(800, 600)
        top_win.show()

        app.exec()


class AnimadorWindow(QMainWindow):
    """Ventana dedicada al animador (texto grande con estilo)."""

    def __init__(self, animator: Animator) -> None:
        super().__init__()
        self.animator = animator
        self.setWindowTitle("Animador")

        self.label = QLabel("Preparando al animador...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("font-size: 36px; font-weight: bold;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.current_text = ""
        self.target_text = ""
        self.char_index = 0

        self.pull_timer = QTimer(self)
        self.pull_timer.timeout.connect(self._pull)
        self.pull_timer.start(1500)

        self.type_timer = QTimer(self)
        self.type_timer.timeout.connect(self._type)

    def _pull(self) -> None:
        if not self.animator.commentary:
            return
        if self.animator.commentary != self.target_text:
            self.target_text = self.animator.commentary
            self.current_text = ""
            self.char_index = 0
            self.type_timer.start(20)

    def _type(self) -> None:
        if self.char_index < len(self.target_text):
            self.current_text += self.target_text[self.char_index]
            self.label.setText(self.current_text)
            self.char_index += 1
        else:
            self.type_timer.stop()


class TopEmotionWindow(QMainWindow):
    """Ventana que muestra a las personas con emociones más intensas (threshold_top)."""

    def __init__(self, animator: Animator) -> None:
        super().__init__()
        self.animator = animator
        self.setWindowTitle("Top emociones del momento")

        # Scroll + contenedor con grilla para 3 filas
        scroll = QScrollArea(self)
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setContentsMargins(12, 12, 12, 12)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(16)
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        # Timer para refrescar la UI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(2000)

    def update_ui(self) -> None:
        data = self.animator.latest_data
        threshold_top = (data or {}).get("threshold_top") or {}
        if not isinstance(threshold_top, dict) or not threshold_top:
            # limpiar si no hay datos
            while self.grid.count():
                item = self.grid.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)
            return

        # Limpiar grilla
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # Colocar tarjetas a lo largo en 3 filas
        idx = 0
        for emotion, top in threshold_top.items():
            if not isinstance(top, TopEmotional):
                continue

            pixmap = self._np_to_qpixmap(top.frame)

            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if pixmap:
                img_lbl.setPixmap(
                    pixmap.scaledToWidth(
                        320, Qt.TransformationMode.SmoothTransformation
                    )
                )
            else:
                img_lbl.setText("Imagen no disponible")

            caption = QLabel(
                f"Persona {top.person_id} → {top.emotion} ({top.score:.2f})"
            )
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setStyleSheet("font-size: 18px; font-weight: bold;")

            card = QWidget()
            v = QVBoxLayout(card)
            v.setContentsMargins(6, 6, 6, 6)
            v.addWidget(caption)
            v.addWidget(img_lbl)

            row = idx % 3
            col = idx // 3
            self.grid.addWidget(card, row, col)
            idx += 1

    def _np_to_qpixmap(self, frame: np.ndarray) -> QPixmap | None:
        """Convierte un np.ndarray (BGR de OpenCV) en QPixmap."""
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            return QPixmap.fromImage(qimg)
        except Exception as e:
            print("Error convirtiendo frame a pixmap:", e)
            return None
