from flask import Flask, jsonify, Response
import cv2
import threading
import subprocess

from robot_controller2 import RobotController

app = Flask(__name__)

robot_running = False

ev3_process = None
controller = RobotController()


@app.get("/")
def home():
    return """
    <h1>EV3 Robot</h1>

    <button onclick="fetch('/start', {method:'POST'})">
        Start
    </button>

    <button onclick="fetch('/stop', {method:'POST'})">
        Stop
    </button>

    <button onclick="status()">
        Status
    </button>

    <pre id="out"></pre>

    <img src="/video" width ="640">

    <script>
    async function status(){
        const r = await fetch("/status");
        document.getElementById("out").innerText =
            JSON.stringify(await r.json(), null, 2);
    }

    setInterval(status,1000);
    status();
    </script>
    """


@app.post("/start")
def start():
    global ev3_process

    # Start EV3 server through SSH
    ev3_process = subprocess.Popen([
        "ssh",
        "robot@10.42.0.3",
        "python3",
        "/home/robot/ev3server/drive_server.py"
    ])

    # Start Raspberry Pi client
    controller.run()

    return "Started"

@app.post("/stop")
def stop():
    global ev3_process

    controller.running = False

    if ev3_process:
        ev3_process.terminate()
        ev3_process = None

    return "Stopped"


@app.get("/status")
def status():

    return jsonify({
        "running": robot_running
    })

def generate_frames():
    while True:
        frame = controller.latest_frame

        if frame is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

@app.get("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

app.run(host="0.0.0.0", port=5050)