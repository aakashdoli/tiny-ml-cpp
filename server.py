#!/usr/bin/env python3
"""
Lightweight Web Server & C++ Execution Bridge for tiny-ml-cpp.
Runs using only the Python standard library.
"""

import http.server
import json
import os
import subprocess
import time
import urllib.parse
from pathlib import Path

PORT = 8080
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
BUILD_DIR = BASE_DIR / "build"
DATA_DIR = BASE_DIR / "data"

def ensure_binaries():
    """Ensure C++ executables are compiled."""
    house_bin = BUILD_DIR / "house_prices"
    knn_bin = BUILD_DIR / "classify_points"
    test_bin = BUILD_DIR / "tinyml_tests"
    
    if not (house_bin.exists() and knn_bin.exists() and test_bin.exists()):
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        cmd = (
            "clang++ -std=c++17 -O3 -Iinclude src/*.cpp examples/house_prices.cpp -o build/house_prices && "
            "clang++ -std=c++17 -O3 -Iinclude src/*.cpp examples/classify_points.cpp -o build/classify_points && "
            "clang++ -std=c++17 -O3 -Iinclude src/*.cpp tests/test_main.cpp -o build/tinyml_tests"
        )
        subprocess.run(cmd, shell=True, cwd=BASE_DIR, check=True)

class TinyMLRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self.handle_api(path, parsed.query)
        else:
            super().do_GET()

    def handle_api(self, path, query):
        if path == "/api/status":
            self.send_json({
                "status": "online",
                "project": "tiny-ml-cpp",
                "has_binaries": (BUILD_DIR / "house_prices").exists(),
            })
        elif path == "/api/run/house_prices":
            self.execute_binary("./build/house_prices")
        elif path == "/api/run/classify_points":
            self.execute_binary("./build/classify_points")
        elif path == "/api/run/tests":
            self.execute_binary("./build/tinyml_tests")
        elif path == "/api/data/houses":
            self.serve_csv_as_json(DATA_DIR / "houses.csv")
        elif path == "/api/data/points":
            self.serve_csv_as_json(DATA_DIR / "points.csv")
        else:
            self.send_error(404, "API endpoint not found")

    def execute_binary(self, binary_cmd):
        ensure_binaries()
        start = time.perf_counter()
        try:
            res = subprocess.run(
                binary_cmd,
                shell=True,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=10
            )
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.send_json({
                "success": res.returncode == 0,
                "command": binary_cmd,
                "output": res.stdout,
                "error": res.stderr,
                "exit_code": res.returncode,
                "duration_ms": duration_ms
            })
        except Exception as e:
            self.send_json({
                "success": False,
                "command": binary_cmd,
                "output": "",
                "error": str(e),
                "duration_ms": 0
            }, status=500)

    def serve_csv_as_json(self, csv_path):
        if not csv_path.exists():
            self.send_error(404, "CSV file not found")
            return
        
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                self.send_json([])
                return
            headers = [h.strip() for h in lines[0].split(",")]
            for line in lines[1:]:
                vals = [float(v.strip()) for v in line.split(",")]
                rows.append(dict(zip(headers, vals)))
        self.send_json(rows)

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def main():
    ensure_binaries()
    port = PORT
    for attempt in range(5):
        try:
            server = http.server.ThreadingHTTPServer(("", port), TinyMLRequestHandler)
            print(f"✨ tiny-ml-cpp visualizer running at http://localhost:{port}")
            print("Press Ctrl+C to stop.")
            server.serve_forever()
            break
        except OSError:
            port += 1

if __name__ == "__main__":
    main()
