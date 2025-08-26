import sys
import time
import threading
from typing import Dict, Any

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer, Qt
from openai import OpenAI

from core.Interpreter import Interpreter


class Animator(threading.Thread):
    def __init__(self, interpreter: Interpreter):
        super().__init__(daemon=True)
        self.interpreter = interpreter
        self.chatgpt_client = OpenAI()
        self.latest_data: Dict[str, Any] = {}
        self.commentary: str = ""

    def run(self) -> None:
        while True:
            time.sleep(15)
            data = {
                "threshold_top": self.interpreter.threshold_top or {},
                "avg_emotion": self.interpreter.avg_persons_emotion or {},
                "num_persons": len(self.interpreter.persons_by_id or {}),
                # "random_person_image": self.interpreter.get_random_person_image() or None,
            }
            self.latest_data = data
            print("Animator snapshot:", data)

            # Generar comentario usando ChatGPT (cliente oficial de OpenAI, modelo hardcodeado)
            try:
                prompt = self._build_prompt(data)
                print("Prompt para ChatGPT:", prompt)

                response = self.chatgpt_client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[
                        {"role": "system", "content": "Eres un animador de eventos."},
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
