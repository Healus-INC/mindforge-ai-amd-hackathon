#!/usr/bin/env python3
"""
Launch MindForge Gradio UI from repo root (use this instead of a root-level `app.py`,
which would shadow the `app/` package on import).

  python run_demo.py

Public URL options:
  • Same machine / LAN: http://127.0.0.1:7860
  • Internet (droplet IP): open inbound TCP 7860 in cloud firewall, then:
      http://YOUR_PUBLIC_IP:7860
    (Example: http://129.212.177.83:7860 — replace with your live public IP if it changes.)
  • Temporary tunnel (judges / quick demo): GRADIO_SHARE=1 python run_demo.py
    (prints a *.gradio.live link; tunnel expires when the process stops.)
"""

from __future__ import annotations

from app.gradio_app_minimal import launch

if __name__ == "__main__":
    launch()
