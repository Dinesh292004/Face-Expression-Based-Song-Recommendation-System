import os, random, logging, time, threading
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)
SONGS_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "songs")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}
DEFAULT_COOLDOWN = 8

class SongRecommender:
    def __init__(self, cooldown_seconds=DEFAULT_COOLDOWN):
        self._cooldown = cooldown_seconds
        self._playlists: Dict[str, List[str]] = {}
        self._current_emotion = None
        self._current_song    = None
        self._last_switch     = 0.0
        self._songs_played    = 0
        self._is_playing      = False
        self._pygame_ok       = False
        self._lock            = threading.Lock()
        self._init_pygame()
        self._load_playlists()

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
            self._pygame_ok = True
        except Exception as e:
            logger.warning("pygame not available: %s", e)

    def _load_playlists(self):
        for emotion in ["happy","sad","angry","neutral","surprise","fear","disgust","contempt","excited","calm","bored"]:
            folder = os.path.join(SONGS_BASE_DIR, emotion)
            os.makedirs(folder, exist_ok=True)
            songs = [os.path.join(folder, f) for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS]
            self._playlists[emotion.capitalize()] = songs
            logger.info("Playlist '%s': %d songs.", emotion, len(songs))

    def on_emotion_detected(self, emotion, confidence):
        with self._lock:
            now = time.time()
            emotion = emotion.capitalize()
            if emotion == self._current_emotion and self._is_playing:
                return None
            if emotion != self._current_emotion:
                if now - self._last_switch < self._cooldown:
                    return None
            song = self._pick_song(emotion)
            if song:
                self._play(song)
                self._current_emotion = emotion
                self._current_song    = song
                self._last_switch     = now
                self._songs_played   += 1
                return song
            return None

    def play_pause(self):
        if not self._pygame_ok: return
        import pygame
        if self._is_playing:
            pygame.mixer.music.pause(); self._is_playing = False
        else:
            pygame.mixer.music.unpause(); self._is_playing = True

    def next_song(self):
        with self._lock:
            if self._current_emotion:
                song = self._pick_song(self._current_emotion)
                if song:
                    self._play(song)
                    self._current_song  = song
                    self._last_switch   = time.time()
                    self._songs_played += 1

    def stop(self):
        if self._pygame_ok:
            import pygame; pygame.mixer.music.stop()
        self._is_playing = False
        self._current_song = None
        self._current_emotion = None

    def reload_playlists(self):
        self._load_playlists()

    def get_playlist(self, emotion):
        return [os.path.basename(s) for s in self._playlists.get(emotion.capitalize(), [])]

    def get_all_playlists(self):
        return {k: [os.path.basename(s) for s in v] for k, v in self._playlists.items()}

    @property
    def current_song_name(self):
        return os.path.basename(self._current_song) if self._current_song else "No song playing"

    @property
    def current_emotion(self): return self._current_emotion

    @property
    def is_playing(self): return self._is_playing

    @property
    def songs_played_count(self): return self._songs_played

    def _pick_song(self, emotion):
        pl = self._playlists.get(emotion.capitalize(), [])
        if not pl: return None
        candidates = [s for s in pl if s != self._current_song] or pl
        return random.choice(candidates)

    def _play(self, path):
        if not self._pygame_ok:
            self._is_playing = True; return
        try:
            import pygame
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(0.75)
            pygame.mixer.music.play()
            self._is_playing = True
        except Exception as e:
            logger.error("Playback error: %s", e)
            self._is_playing = False
