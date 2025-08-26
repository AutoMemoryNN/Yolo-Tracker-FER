import sys
import time
import threading
from typing import Dict, Any

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer

from core.Interpreter import Interpreter


class Animator(threading.Thread):
    def __init__(self, interpreter: Interpreter):
        super().__init__(daemon=True)
        self.interpreter = interpreter
        self.latest_data: Dict[str, Any] = {}

    def run(self) -> None:
        while True:
            time.sleep(5)
            data = {
                "threshold_top": self.interpreter.threshold_top or {},
                "avg_emotion": self.interpreter.avg_persons_emotion or {},
                "num_persons": len(self.interpreter.persons_by_id or {}),
            }
            self.latest_data = data
            print("Animator snapshot:", data)

    def start_windows(self) -> None:
        app = QApplication(sys.argv)

        t_interp = threading.Thread(target=self.interpreter.interpret, daemon=True)
        t_interp.start()

        animator = Animator(self.interpreter)
        animator.start()

        win1 = MyWindow("Ventana 1", animator)
        win2 = MyWindow("Ventana 2", animator)
        win1.show()
        win2.show()

        sys.exit(app.exec())


class MyWindow(QMainWindow):
    def __init__(self, nombre: str, animator: Animator) -> None:
        super().__init__()
        self.animator = animator
        self.setWindowTitle(nombre)
        self.label = QLabel("Esperando datos...")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Refrescar UI cada 2s
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(2000)

    def update_ui(self) -> None:
        data = self.animator.latest_data
        if not data:
            return
        txt = f"{self.windowTitle()} -> Personas: {data['num_persons']}, Avg: {data['avg_emotion']}, Top: {data['threshold_top'].keys()}"
        self.label.setText(txt)
