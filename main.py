"""
main.py — application entry point.

Startup sequence:
  1. Create QApplication
  2. Build MainWindow (which kicks off background pre-training)
  3. Show window
  4. Enter Qt event loop

Run from project root:
    python main.py
"""

import sys
import os

# ── ensure project root is on sys.path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from PyQt6.QtGui     import QFont

from ui.main_window import MainWindow


def main():
    # High-DPI is enabled by default in PyQt6 — no attribute needed
    app = QApplication(sys.argv)

    # Application-wide default font
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("Visual Neural Network")
    app.setOrganizationName("NSD Demo")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
