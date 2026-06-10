from machine import Pin
from time import sleep, ticks_ms, ticks_diff
import network
import socket
import os
import gc

try:
    import updater
except Exception as e:
    updater = None
    print("Updater not available:", e)


# ============================================================
# Konfiguration
# ============================================================

LED_PINS = [14, 27, 25, 32, 33]

LOG_FILE = "recordings.csv"

# Sehr kleiner Timeout, damit update_blinking() regelmäßig laufen kann.
SOCKET_TIMEOUT_SECONDS = 0.05

# Fehlercode-Definitionen
# states: Startzustand der LEDs
# blink_ms: 0 = kein Blinken, sonst Blinkintervall in Millisekunden
ERROR_CODES = {
    "fehlercode_01": {
        "description": "Alle LEDs aus",
        "states": [0, 0, 0, 0, 0],
        "blink_ms": [0, 0, 0, 0, 0]
    },
    "fehlercode_02": {
        "description": "LED 1 und LED 5 dauerhaft an",
        "states": [1, 0, 0, 0, 1],
        "blink_ms": [0, 0, 0, 0, 0]
    },
    "fehlercode_03": {
        "description": "LED 2 blinkt langsam, LED 4 und LED 5 an",
        "states": [0, 1, 0, 1, 1],
        "blink_ms": [0, 700, 0, 0, 0]
    },
    "fehlercode_04": {
        "description": "LED 3 blinkt schnell, LED 1 an",
        "states": [1, 0, 1, 0, 0],
        "blink_ms": [0, 0, 300, 0, 0]
    },
    "fehlercode_05": {
        "description": "LED 1, LED 2 und LED 3 blinken gemeinsam",
        "states": [1, 1, 1, 0, 0],
        "blink_ms": [500, 500, 500, 0, 0]
    }
}

ERROR_CODE_ORDER = [
    "fehlercode_01",
    "fehlercode_02",
    "fehlercode_03",
    "fehlercode_04",
    "fehlercode_05"
]


# ============================================================
# Globale Zustände
# ============================================================

leds = []
led_states = [0, 0, 0, 0, 0]

active_error_code = None
active_base_states = [0, 0, 0, 0, 0]
active_blink_ms = [0, 0, 0, 0, 0]
last_blink_toggle = [0, 0, 0, 0, 0]


# ============================================================
# LED-Steuerung
# ============================================================

def setup_leds():
    global leds

    leds = []

    for pin in LED_PINS:
        led = Pin(pin, Pin.OUT)
        led.off()
        leds.append(led)

    print("LEDs initialized")


def internal_set_led(index, state):
    if index < 0 or index >= len(leds):
        return False

    if state == 1:
        leds[index].on()
        led_states[index] = 1
    else:
        leds[index].off()
        led_states[index] = 0

    return True


def stop_error_code(also_turn_off=False):
    global active_error_code
    global active_base_states
    global active_blink_ms

    active_error_code = None
    active_base_states = [0, 0, 0, 0, 0]
    active_blink_ms = [0, 0, 0, 0, 0]

    if also_turn_off:
        for i in range(len(leds)):
            internal_set_led(i, 0)

    print("Error code stopped")


def set_led(index, state):
    stop_error_code(False)

    success = internal_set_led(index, state)

    if success:
        print("LED", index + 1, "set to", state)

    return success


def set_all_leds(state):
    stop_error_code(False)

    for i in range(len(leds)):
        internal_set_led(i, state)

    print("All LEDs set to", state)

def set_led_bits(bits):
    if bits is None:
        return False

    if len(bits) != len(leds):
        return False

    for char in bits:
        if char not in ("0", "1"):
            return False

    stop_error_code(False)

    for i in range(len(leds)):
        state = 1 if bits[i] == "1" else 0
        internal_set_led(i, state)

    print("LED bits set to:", bits)

    return True


def run_test_pattern():
    stop_error_code(False)

    print("Running test pattern")

    for i in range(len(leds)):
        internal_set_led(i, 1)
        sleep(0.2)
        internal_set_led(i, 0)

    sleep(0.3)

    for i in range(len(leds)):
        internal_set_led(i, 1)

    sleep(0.7)

    for i in range(len(leds)):
        internal_set_led(i, 0)


def play_error_code(code):
    global active_error_code
    global active_base_states
    global active_blink_ms
    global last_blink_toggle

    if code not in ERROR_CODES:
        return False

    definition = ERROR_CODES[code]

    active_error_code = code
    active_base_states = definition["states"][:]
    active_blink_ms = definition["blink_ms"][:]

    now = ticks_ms()
    last_blink_toggle = [now, now, now, now, now]

    for i in range(len(leds)):
        internal_set_led(i, active_base_states[i])

    print("Playing error code:", code)

    return True


def update_blinking():
    if active_error_code is None:
        return

    now = ticks_ms()

    for i in range(len(leds)):
        interval = active_blink_ms[i]

        if interval > 0:
            if ticks_diff(now, last_blink_toggle[i]) >= interval:
                if led_states[i] == 1:
                    internal_set_led(i, 0)
                else:
                    internal_set_led(i, 1)

                last_blink_toggle[i] = now


# ============================================================
# Netzwerk
# ============================================================

def get_ip_address():
    wlan = network.WLAN(network.STA_IF)

    if wlan.isconnected():
        return wlan.ifconfig()[0]

    return "not connected"


def check_update_on_startup():
    if updater is None:
        print("Startup update check skipped: updater not available")
        return

    print("Startup update check...")

    try:
        gc.collect()
        print("Free memory before update check:", gc.mem_free())
        updater.check_for_update()
        gc.collect()
        print("Free memory after update check:", gc.mem_free())

    except Exception as e:
        print("Startup update check failed:", e)
        print("System continues with existing version")

    finally:
        gc.collect()


# ============================================================
# Hilfsfunktionen
# ============================================================

def file_exists(path):
    try:
        os.stat(path)
        return True
    except Exception:
        return False


def ensure_log_file():
    if not file_exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write(
                "created_ms,file_name,error_code,environment,"
                "roi_x,roi_y,roi_w,roi_h,"
                "lighting,camera_position,distance_cm,notes\n"
            )


def url_decode(value):
    if value is None:
        return ""

    value = value.replace("+", " ")

    result = ""
    i = 0

    while i < len(value):
        if value[i] == "%" and i + 2 < len(value):
            try:
                result += chr(int(value[i + 1:i + 3], 16))
                i += 3
            except Exception:
                result += value[i]
                i += 1
        else:
            result += value[i]
            i += 1

    return result


def json_escape(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")

    return value


def csv_escape(value):
    if value is None:
        value = ""

    value = str(value)
    value = value.replace('"', '""')

    if "," in value or '"' in value or "\n" in value or "\r" in value:
        return '"' + value + '"'

    return value


def parse_request_path(request):
    try:
        first_line = request.split("\r\n")[0]
        parts = first_line.split(" ")

        if len(parts) >= 2:
            return parts[1]

    except Exception as e:
        print("Request parse error:", e)

    return "/"


def parse_query(path):
    result = {}

    if "?" not in path:
        return result

    query_string = path.split("?", 1)[1]
    pairs = query_string.split("&")

    for pair in pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key] = url_decode(value)
        else:
            result[pair] = ""

    return result


def append_log_entry(data):
    ensure_log_file()

    row = [
        str(ticks_ms()),
        data.get("file", ""),
        data.get("code", ""),
        data.get("env", ""),
        data.get("roi_x", ""),
        data.get("roi_y", ""),
        data.get("roi_w", ""),
        data.get("roi_h", ""),
        data.get("lighting", ""),
        data.get("camera", ""),
        data.get("distance", ""),
        data.get("notes", "")
    ]

    line = ",".join([csv_escape(item) for item in row]) + "\n"

    with open(LOG_FILE, "a") as f:
        f.write(line)

    print("Log entry saved")

    return True


def read_log_file():
    ensure_log_file()

    with open(LOG_FILE, "r") as f:
        return f.read()


def clear_log_file():
    if file_exists(LOG_FILE):
        os.remove(LOG_FILE)

    ensure_log_file()


# ============================================================
# JSON-Antworten
# ============================================================

def json_status():
    leds_json = ",".join([str(state) for state in led_states])

    if active_error_code is None:
        active_code_json = "null"
    else:
        active_code_json = '"' + json_escape(active_error_code) + '"'

    return (
        "{"
        '"leds":[' + leds_json + "],"
        '"active_error_code":' + active_code_json + ","
        '"free_memory":' + str(gc.mem_free()) +
        "}"
    )


def json_codes():
    parts = []

    for code in ERROR_CODE_ORDER:
        definition = ERROR_CODES[code]

        item = (
            "{"
            '"code":"' + json_escape(code) + '",'
            '"description":"' + json_escape(definition["description"]) + '"'
            "}"
        )

        parts.append(item)

    return '{"codes":[' + ",".join(parts) + "]}"


# ============================================================
# HTTP Response
# ============================================================

def send_all(client, data):
    chunk_size = 512

    for i in range(0, len(data), chunk_size):
        client.send(data[i:i + chunk_size])


def send_response(client, body, content_type="application/json", status="200 OK"):
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    else:
        body_bytes = body

    header = (
        "HTTP/1.1 " + status + "\r\n"
        "Content-Type: " + content_type + "\r\n"
        "Content-Length: " + str(len(body_bytes)) + "\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n"
    )

    send_all(client, header.encode("utf-8"))
    send_all(client, body_bytes)


# ============================================================
# API Routing
# ============================================================

def handle_request(path):
    print("Request:", path)

    if path == "/" or path.startswith("/?"):
        return (
            "ESP32 LED API is running.\n"
            "Use /api/status, /api/codes, /api/error?code=fehlercode_03\n",
            "text/plain",
            "200 OK"
        )

    if path.startswith("/api/status"):
        return json_status(), "application/json", "200 OK"

    if path.startswith("/api/codes"):
        return json_codes(), "application/json", "200 OK"

    if path.startswith("/api/state"):
        query = parse_query(path)
        bits = query.get("bits", "")

        success = set_led_bits(bits)

        if success:
            return json_status(), "application/json", "200 OK"

        return '{"ok":false,"error":"invalid bits"}', "application/json", "400 Bad Request"

    if path.startswith("/api/set"):
        query = parse_query(path)

        try:
            led_number = int(query.get("led", "0"))
            state = int(query.get("state", "0"))

            led_index = led_number - 1

            success = set_led(led_index, state)

            if success:
                return json_status(), "application/json", "200 OK"

            return '{"ok":false,"error":"invalid led index"}', "application/json", "400 Bad Request"

        except Exception as e:
            print("Set LED error:", e)
            return '{"ok":false,"error":"invalid request"}', "application/json", "400 Bad Request"

    if path.startswith("/api/all"):
        query = parse_query(path)

        try:
            state = int(query.get("state", "0"))
            set_all_leds(state)
            return json_status(), "application/json", "200 OK"

        except Exception as e:
            print("Set all LEDs error:", e)
            return '{"ok":false,"error":"invalid request"}', "application/json", "400 Bad Request"

    if path.startswith("/api/pattern"):
        run_test_pattern()
        return json_status(), "application/json", "200 OK"

    if path.startswith("/api/error"):
        query = parse_query(path)
        code = query.get("code", "")

        success = play_error_code(code)

        if success:
            return json_status(), "application/json", "200 OK"

        return '{"ok":false,"error":"unknown error code"}', "application/json", "400 Bad Request"

    if path.startswith("/api/stop"):
        stop_error_code(True)
        return json_status(), "application/json", "200 OK"

    if path.startswith("/api/log?"):
        query = parse_query(path)

        try:
            append_log_entry(query)
            return '{"ok":true}', "application/json", "200 OK"

        except Exception as e:
            print("Log save error:", e)
            return '{"ok":false,"error":"log save failed"}', "application/json", "500 Internal Server Error"

    if path.startswith("/api/logs"):
        try:
            return read_log_file(), "text/csv", "200 OK"

        except Exception as e:
            print("Log read error:", e)
            return "error", "text/plain", "500 Internal Server Error"

    if path.startswith("/api/clear-log"):
        try:
            clear_log_file()
            return '{"ok":true}', "application/json", "200 OK"

        except Exception as e:
            print("Log clear error:", e)
            return '{"ok":false,"error":"log clear failed"}', "application/json", "500 Internal Server Error"

    return "Not found", "text/plain", "404 Not Found"


# ============================================================
# Webserver
# ============================================================

def start_server():
    ip = get_ip_address()

    print("ESP32 IP address:", ip)
    print("API base URL: http://" + ip)

    address = socket.getaddrinfo("0.0.0.0", 80)[0][-1]

    server_socket = socket.socket()

    try:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass

    server_socket.bind(address)
    server_socket.listen(3)
    server_socket.settimeout(SOCKET_TIMEOUT_SECONDS)

    print("Webserver started on port 80")

    return server_socket


# ============================================================
# Main Loop
# ============================================================

def main():
    setup_leds()
    ensure_log_file()

    print("LED API controller started")
    print("No camera frontend on ESP32")
    print("Startup update check enabled")
    print("No periodic HTTPS updater during runtime")

    # Einmaliger Update-Check beim Start.
    # Danach wird kein periodischer Update-Check mehr ausgefuehrt,
    # damit der ESP32 waehrend der Datenerzeugung stabil bleibt.
    check_update_on_startup()

    server_socket = start_server()

    while True:
        update_blinking()

        try:
            client, client_address = server_socket.accept()

            try:
                request = client.recv(2048).decode("utf-8")
                path = parse_request_path(request)

                # CORS Preflight
                if request.startswith("OPTIONS"):
                    send_response(client, "", "text/plain", "204 No Content")
                else:
                    body, content_type, status = handle_request(path)
                    send_response(client, body, content_type, status)

            except Exception as e:
                print("Client handling error:", e)

            finally:
                try:
                    client.close()
                except Exception:
                    pass

                gc.collect()
                print("Free memory:", gc.mem_free())

        except OSError:
            # normaler Timeout, damit update_blinking weiterlaufen kann
            pass

        except Exception as e:
            print("Server error:", e)
            sleep(0.2)
            gc.collect()


main()
