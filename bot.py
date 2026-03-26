import asyncio
import os
import re
import time
import keyboard
import discord
import edge_tts
import mss
import pytesseract
from PIL import Image

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    from dotenv import load_dotenv
    load_dotenv()
    TOKEN = os.getenv("BOT_TOKEN")
GUILD_ID = 143166145350860800
VOICE_CHANNEL_ID = 388138351238316035
COMMAND_CHANNEL_ID = 1147970850260254740

TTS_VOICE = "en-GB-SoniaNeural"
SCAN_INTERVAL = 10  # seconds between automatic scans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = os.path.join(BASE_DIR, "third_party", "ffmpeg.exe")

# ---------------------------------------------------------------------------
# Overview table layout (absolute screen coordinates)
# ---------------------------------------------------------------------------
# The capture region that covers all rows and columns
OVERVIEW_LEFT = 1494
OVERVIEW_TOP = 278
OVERVIEW_RIGHT = 1798
OVERVIEW_BOTTOM = 682

ROW_HEIGHT = 28     # pixels per row
CELL_HEIGHT = 20    # readable text area within a row

# First row starts at y=279 absolute → 1 pixel into the capture
FIRST_ROW_Y = 279 - OVERVIEW_TOP  # relative y offset

# Column boundaries (absolute x → converted to relative in capture)
COLUMNS = {
    "ship_type": (1506 - OVERVIEW_LEFT, 1609 - OVERVIEW_LEFT),
    "distance":  (1617 - OVERVIEW_LEFT, 1686 - OVERVIEW_LEFT),
    "velocity":  (1742 - OVERVIEW_LEFT, 1793 - OVERVIEW_LEFT),
}

MAX_ROWS = (OVERVIEW_BOTTOM - 279) // ROW_HEIGHT  # ~14
# ---------------------------------------------------------------------------

SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

intents = discord.Intents.default()
intents.voice_states = True
intents.guilds = True
intents.message_content = True

client = discord.Client(intents=intents)
voice_client: discord.VoiceClient | None = None
scan_task: asyncio.Task | None = None
active = False

popup_mode = False
popup_task: asyncio.Task | None = None
_prev_ship_count = 0

POPUP_SCAN_INTERVAL = 3  # seconds between popup-mode scans

_last_toggle = 0.0


MIN_CONFIDENCE = 50       # tesseract confidence threshold (0-100)
MIN_SHIP_NAME_LEN = 3    # reject ship names shorter than this
VARIANCE_THRESHOLD = 200  # minimum pixel variance to attempt OCR


def _cell_has_content(cell: Image.Image) -> bool:
    """Check if a cell has enough contrast to plausibly contain text."""
    grayscale = cell.convert("L")
    pixels = list(grayscale.getdata())
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    return variance > VARIANCE_THRESHOLD


def _ocr_cell(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> str:
    """Crop a cell, check for content, upscale, and OCR with confidence filtering."""
    cell = img.crop((x1, y1, x2, y2))

    if not _cell_has_content(cell):
        return ""

    cell = cell.resize((cell.width * 4, cell.height * 4), Image.LANCZOS)
    data = pytesseract.image_to_data(cell, config="--psm 7", output_type=pytesseract.Output.DICT)

    words = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = int(data["conf"][i])
        if word and conf >= MIN_CONFIDENCE:
            words.append(word)

    return " ".join(words)


def capture_ship_table() -> list[dict]:
    """Screenshot the overview area and OCR each row's columns."""
    monitor = {
        "left": OVERVIEW_LEFT,
        "top": OVERVIEW_TOP,
        "width": OVERVIEW_RIGHT - OVERVIEW_LEFT,
        "height": OVERVIEW_BOTTOM - OVERVIEW_TOP,
    }

    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    ships = []
    for row in range(MAX_ROWS):
        y_top = FIRST_ROW_Y + row * ROW_HEIGHT
        y_bottom = y_top + CELL_HEIGHT

        if y_bottom > img.height:
            break

        ship_x1, ship_x2 = COLUMNS["ship_type"]
        ship_type = _ocr_cell(img, ship_x1, y_top, ship_x2, y_bottom)

        if not ship_type or len(ship_type) < MIN_SHIP_NAME_LEN:
            break

        dist_x1, dist_x2 = COLUMNS["distance"]
        distance_raw = _ocr_cell(img, dist_x1, y_top, dist_x2, y_bottom)

        vel_x1, vel_x2 = COLUMNS["velocity"]
        velocity_raw = _ocr_cell(img, vel_x1, y_top, vel_x2, y_bottom)

        ships.append({
            "ship_type": ship_type,
            "distance": distance_raw,
            "velocity": velocity_raw,
        })

    return ships


def _parse_distance(raw: str) -> str:
    """Turn '499 km' or '499' into '499 kilometers'."""
    nums = re.findall(r"[\d.]+", raw)
    if not nums:
        return raw or "unknown distance"
    return f"{nums[0]} kilometers"


def _parse_velocity(raw: str) -> str:
    """Turn '140' into '140 meters per second'."""
    nums = re.findall(r"[\d.]+", raw)
    if not nums:
        return raw or "unknown speed"
    val = nums[0]
    if val == "0":
        return "stationary"
    return f"{val} meters per second"


def _speed_category(raw: str) -> str:
    """Return 'hot' if velocity > 300 m/s, otherwise 'cold'."""
    nums = re.findall(r"[\d.]+", raw)
    if not nums:
        return "cold"
    try:
        return "hot" if float(nums[0]) > 300 else "cold"
    except ValueError:
        return "cold"


def format_popup_announcement(new_ships: list[dict]) -> str:
    """Build the AWACS pop-up call for newly detected contacts."""
    parts = []
    for s in new_ships:
        name = s["ship_type"]
        dist = _parse_distance(s["distance"])
        speed = _speed_category(s["velocity"])
        parts.append(
            f"SATAN 1, Magic, Pop-up group, {name}, {dist}, {speed}"
        )
    return ". ".join(parts)


def format_announcement(ships: list[dict]) -> str:
    """Build a TTS-friendly string from the ship table."""
    parts = []
    for s in ships:
        name = s["ship_type"]
        dist = _parse_distance(s["distance"])
        vel = _parse_velocity(s["velocity"])
        parts.append(f"{name}, {dist}, {vel}")
    return ". ".join(parts)


async def generate_tts(text: str) -> str:
    path = os.path.join(TEMP_DIR, "tts_output.mp3")
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(path)
    return path


async def play_file(vc: discord.VoiceClient, path: str):
    if vc.is_playing():
        vc.stop()
    source = discord.FFmpegPCMAudio(path, executable=FFMPEG_PATH)
    vc.play(source)
    while vc.is_playing():
        await asyncio.sleep(0.1)


async def play_sound(vc: discord.VoiceClient, filename: str):
    path = os.path.join(SOUNDS_DIR, filename)
    if not os.path.isfile(path):
        print(f"[WARN] Sound file not found: {path}")
        return
    await play_file(vc, path)


async def ensure_connected() -> discord.VoiceClient | None:
    """Return a connected VoiceClient, reconnecting if needed."""
    global voice_client

    if voice_client and voice_client.is_connected():
        return voice_client

    channel = client.get_channel(VOICE_CHANNEL_ID)
    if channel is None:
        print(f"[ERROR] Could not find voice channel {VOICE_CHANNEL_ID}")
        return None

    if voice_client:
        try:
            await voice_client.disconnect(force=True)
        except Exception:
            pass
        voice_client = None

    try:
        print("[RECONNECT] Joining voice channel...")
        voice_client = await channel.connect(self_deaf=True, timeout=15.0)
        print("[RECONNECT] Connected.")
        return voice_client
    except Exception as e:
        print(f"[RECONNECT] Failed: {e}")
        return None


async def announce_ships(vc: discord.VoiceClient) -> bool:
    """Capture the overview table, format it, and read it aloud. Returns True if ships found."""
    print("[SCAN] Capturing overview table...")
    ships = capture_ship_table()

    if not ships:
        print("[SCAN] No ships detected.")
        return False

    for s in ships:
        print(f"  {s['ship_type']:>20s}  |  {s['distance']:>10s}  |  {s['velocity']:>6s}")

    announcement = format_announcement(ships)
    print(f"[TTS] {announcement}")
    tts_path = await generate_tts(announcement)

    await play_file(vc, tts_path)
    print("[TTS] Done.")
    return True


async def scan_loop():
    """Background loop: scan and announce every SCAN_INTERVAL seconds."""
    await asyncio.sleep(1.0)
    while active:
        try:
            vc = await ensure_connected()
            if vc:
                await announce_ships(vc)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[SCAN LOOP] Error: {e}")
        await asyncio.sleep(SCAN_INTERVAL)


async def popup_scan_loop():
    """Background loop: watch for ship count increases and announce pop-ups."""
    global _prev_ship_count
    await asyncio.sleep(1.0)

    _prev_ship_count = len(capture_ship_table())
    print(f"[POPUP] Baseline: {_prev_ship_count} contact(s)")

    while popup_mode:
        try:
            ships = capture_ship_table()
            current_count = len(ships)

            if current_count > _prev_ship_count:
                new_ships = ships[_prev_ship_count:]
                print(f"[POPUP] +{len(new_ships)} new contact(s) detected!")
                for s in new_ships:
                    print(f"  {s['ship_type']:>20s}  |  {s['distance']:>10s}  |  {s['velocity']:>6s}")

                vc = await ensure_connected()
                if vc:
                    text = format_popup_announcement(new_ships)
                    print(f"[POPUP TTS] {text}")
                    tts_path = await generate_tts(text)
                    await play_file(vc, tts_path)

            _prev_ship_count = current_count
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[POPUP LOOP] Error: {e}")
        await asyncio.sleep(POPUP_SCAN_INTERVAL)


async def bogey_dope():
    """One-shot scan. Joins voice if not already connected."""
    global active
    if not active:
        active = True
    vc = await ensure_connected()
    if vc is None:
        print("[BOGEY DOPE] Could not connect to voice.")
        active = False
        return
    found = await announce_ships(vc)
    if not found:
        print("[BOGEY DOPE] Clean -- announcing.")
        tts_path = await generate_tts("SATAN 1, Magic, picture Clean.")
        await play_file(vc, tts_path)


async def toggle_voice():
    """Join or leave the voice channel. Starts/stops the scan loop."""
    global voice_client, scan_task, active

    if active:
        print("[TOGGLE] Deactivating...")
        active = False
        if scan_task and not scan_task.done():
            scan_task.cancel()
            scan_task = None
        if voice_client:
            try:
                await voice_client.disconnect(force=True)
            except Exception:
                pass
            voice_client = None
        print("[TOGGLE] Disconnected. Scan loop stopped.")
        return

    active = True
    vc = await ensure_connected()
    if vc is None:
        print("[TOGGLE] Could not connect to voice.")
        active = False
        return
    print("[TOGGLE] Active. Scanning every {0}s...".format(SCAN_INTERVAL))
    scan_task = asyncio.create_task(scan_loop())


async def declare_popup():
    """Toggle popup awareness mode on/off."""
    global popup_mode, popup_task, _prev_ship_count

    if popup_mode:
        print("[POPUP] Deactivating popup mode...")
        popup_mode = False
        if popup_task and not popup_task.done():
            popup_task.cancel()
            popup_task = None
        _prev_ship_count = 0
        print("[POPUP] Popup mode off.")
        return

    popup_mode = True
    vc = await ensure_connected()
    if vc is None:
        print("[POPUP] Could not connect to voice.")
        popup_mode = False
        return

    print(f"[POPUP] Popup mode ON. Scanning every {POPUP_SCAN_INTERVAL}s for new contacts...")
    tts_path = await generate_tts("Magic, popup mode active.")
    await play_file(vc, tts_path)
    popup_task = asyncio.create_task(popup_scan_loop())


async def test_beep():
    vc = await ensure_connected()
    if vc is None:
        print("[BEEP] Not connected.")
        return
    print("[BEEP] Playing test beep...")
    await play_sound(vc, "beep.wav")
    print("[BEEP] Done.")


def _schedule(coro):
    asyncio.run_coroutine_threadsafe(coro, client.loop)


def _schedule_debounced_toggle():
    global _last_toggle
    now = time.monotonic()
    if now - _last_toggle < 1.0:
        return
    _last_toggle = now
    _schedule(toggle_voice())


@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print()
    print("  Ctrl+Shift+W  ->  Toggle scan loop (every {0}s)".format(SCAN_INTERVAL))
    print("  Ctrl+Shift+T  ->  Play test beep")
    print()
    print(f'  "scan"            ->  Toggle scan loop       (in #{COMMAND_CHANNEL_ID})')
    print(f'  "bogey dope"      ->  One-shot scan          (in #{COMMAND_CHANNEL_ID})')
    print(f'  "declare popup"   ->  Toggle popup awareness (in #{COMMAND_CHANNEL_ID})')
    print(f'  "cancel popup"    ->  Stop popup mode        (in #{COMMAND_CHANNEL_ID})')
    print()
    print(f"  Overview region: ({OVERVIEW_LEFT},{OVERVIEW_TOP}) -> ({OVERVIEW_RIGHT},{OVERVIEW_BOTTOM})")
    print(f"  Max rows: {MAX_ROWS}, row height: {ROW_HEIGHT}px")
    print(f"  TTS voice: {TTS_VOICE}")
    print()

    keyboard.add_hotkey("ctrl+shift+w", _schedule_debounced_toggle)
    keyboard.add_hotkey("ctrl+shift+t", lambda: _schedule(test_beep()))

    print("Hotkeys registered. Waiting for input...")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot or message.channel.id != COMMAND_CHANNEL_ID:
        return

    text = message.content.strip().lower()

    if text == "scan":
        status = "Deactivating" if active else "Activating"
        print(f"[CMD] '{message.author}' said 'scan' -- {status}...")
        await toggle_voice()

    elif text == "bogey dope":
        print(f"[CMD] '{message.author}' said 'bogey dope' -- one-shot scan...")
        await bogey_dope()

    elif text == "declare popup":
        status = "Deactivating" if popup_mode else "Activating"
        print(f"[CMD] '{message.author}' said 'declare popup' -- {status} popup mode...")
        await declare_popup()

    elif text == "cancel popup":
        if popup_mode:
            print(f"[CMD] '{message.author}' said 'cancel popup' -- stopping...")
            await declare_popup()
        else:
            print("[CMD] 'cancel popup' received but popup mode not active.")


client.run(TOKEN)
