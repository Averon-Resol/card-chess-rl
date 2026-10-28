"""Stockfish-backed move engine for Card-Chess (Revision 2, decision R1).

Why: our pure-Python minimax takes ~80 s per move at depth 6 (measured
0.02 / 0.26 / 1.7 s at depths 2 / 3 / 4), which makes depth-6 training impossible.
Stockfish (C++) searches depth 6 in milliseconds.

How the card rules are kept:
  * Card-legal moves that are also standard chess moves are searched by Stockfish
    directly at `depth`, restricted to exactly those root moves (`root_moves`).
  * Card-added, non-standard moves (e.g. a queen jumping like a knight) cannot be
    sent to Stockfish as moves. We play each one ourselves and ask Stockfish to
    score the resulting position at `depth - 1`, so the total is still `depth` plies.
  * As with the old minimax, plies below the root assume standard chess rules.
    This is a documented approximation.

Stockfish is found via (1) the STOCKFISH_PATH env variable, (2) the project's
`engines/` folder, (3) the system PATH. If none is found on Windows or Linux, it is
downloaded automatically into `engines/`.
"""
from __future__ import annotations

import glob
import io
import math
import os
import platform
import shutil
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import chess
import chess.engine

from cardchess_rl.search.evaluation import evaluate
from cardchess_rl.search.minimax import choose_move as python_choose_move

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = PROJECT_ROOT / "engines"
MATE_SCORE = 100_000  # centipawn value used for "mate in N"

_RELEASE = "https://github.com/official-stockfish/Stockfish/releases/latest/download/"
_ASSETS = {
    "Windows": ["stockfish-windows-x86-64-universal.zip"],
    "Linux": ["stockfish-linux-x86-64-universal.tar.gz"],
    "Darwin": ["stockfish-macos-universal.tar.gz"],
}


# --------------------------------------------------------------------------- #
# Locating / downloading Stockfish
# --------------------------------------------------------------------------- #
def _find_local() -> str | None:
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).is_file():
        return env
    if ENGINE_DIR.exists():
        pattern = "stockfish*.exe" if platform.system() == "Windows" else "stockfish*"
        for path in sorted(glob.glob(str(ENGINE_DIR / "**" / pattern), recursive=True)):
            p = Path(path)
            if p.is_file() and p.suffix not in {".zip", ".tar", ".gz", ".txt", ".md"}:
                return str(p)
    return shutil.which("stockfish")


def _works(path: str) -> bool:
    """Return True if the binary starts and answers UCI (catches CPUs without AVX2)."""
    try:
        eng = chess.engine.SimpleEngine.popen_uci(path)
        eng.quit()
        return True
    except Exception:
        return False


def _download(asset: str) -> str | None:
    ENGINE_DIR.mkdir(exist_ok=True)
    url = _RELEASE + asset
    print(f"[stockfish] Downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "CardChessRL/1.0"})
    data = urllib.request.urlopen(req, timeout=180).read()
    stem = asset.replace(".tar.gz", "").replace(".zip", "").replace(".tar", "")
    target = ENGINE_DIR / stem
    if asset.endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(target)
    else:
        tarfile.open(fileobj=io.BytesIO(data), mode="r:*").extractall(target)
    for path in glob.glob(str(target / "**" / "stockfish*"), recursive=True):
        p = Path(path)
        if p.is_file() and (p.suffix == ".exe" or (platform.system() != "Windows" and p.suffix == "")):
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
            return str(p)
    return None


def ensure_stockfish() -> str:
    """Return a path to a working Stockfish binary, downloading it if needed."""
    path = _find_local()
    if path and _works(path):
        return path
    system = platform.system()
    for asset in _ASSETS.get(system, []):
        try:
            path = _download(asset)
        except Exception as exc:  # network problems etc.
            print(f"[stockfish] Download of {asset} failed: {exc}")
            continue
        if path and _works(path):
            print(f"[stockfish] Ready: {path}")
            return path
    raise RuntimeError(
        "Could not find or download Stockfish.\n"
        "  Windows/Linux: download it from https://stockfishchess.org/download/ and put the "
        f"executable in {ENGINE_DIR}\n"
        "  macOS: run `brew install stockfish`\n"
        "  Or set the STOCKFISH_PATH environment variable to the executable."
    )


# --------------------------------------------------------------------------- #
# Move engine
# --------------------------------------------------------------------------- #
def cp_to_potential(cp: float) -> float:
    """Map a centipawn score to [-1, 1] via the logistic win-probability curve.

    P_win = 1 / (1 + 10^(-cp/400))   (Elo / Lichess convention)
    Phi   = 2*P_win - 1 = tanh(cp * ln(10) / 800)
    """
    return math.tanh(cp * math.log(10) / 800.0)


class StockfishMover:
    """Chooses card-legal moves using Stockfish at a fixed search depth."""

    def __init__(self, depth: int = 6, threads: int = 1, hash_mb: int = 16, path: str | None = None):
        self.depth = depth
        self._engine: chess.engine.SimpleEngine | None = None
        self.path = path or ensure_stockfish()
        self._opts = {"Threads": threads, "Hash": hash_mb}
        self._start()

    # -- process management ------------------------------------------------- #
    def _start(self) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self._engine.configure(self._opts)

    def close(self) -> None:
        if getattr(self, "_engine", None) is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def _analyse(self, board: chess.Board, depth: int, root_moves=None) -> dict:
        # Strip move_stack so python-chess sends `position fen <FEN>` instead of
        # replaying custom card moves from the move history.
        clean = board.copy(stack=False)
        try:
            return self._engine.analyse(clean, chess.engine.Limit(depth=depth), root_moves=root_moves)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
            # Restart once if Stockfish crashed, then retry.
            self.close()
            self._start()
            return self._engine.analyse(clean, chess.engine.Limit(depth=depth), root_moves=root_moves)

    # -- public API ---------------------------------------------------------- #
    def score_cp(self, board: chess.Board, color: chess.Color, depth: int | None = None) -> float:
        """Centipawn evaluation of `board` from `color`'s perspective."""
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == color else MATE_SCORE
        if board.is_game_over():
            return 0.0
        clean = board.copy(stack=False)
        if not clean.is_valid():
            v = evaluate(clean)
            return v if color == chess.WHITE else -v
        try:
            info = self._analyse(clean, depth or self.depth)
            return float(info["score"].pov(color).score(mate_score=MATE_SCORE))
        except Exception:
            v = evaluate(clean)
            return v if color == chess.WHITE else -v

    def choose_move(self, board: chess.Board, legal_moves: list[chess.Move] | None = None) -> chess.Move | None:
        """Best move among `legal_moves` (card-modified) for the side to move."""
        moves = list(legal_moves) if legal_moves is not None else list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        clean = board.copy(stack=False)
        if not clean.is_valid():
            return python_choose_move(board, moves, depth=2)

        me = clean.turn
        std_legal = set(clean.legal_moves)
        standard = [m for m in moves if m in std_legal]
        extra = [m for m in moves if m not in std_legal]

        best_move, best_score = None, -math.inf
        if standard:
            try:
                info = self._analyse(clean, self.depth, root_moves=standard)
                pv = info.get("pv")
                if pv:
                    best_move = pv[0]
                    best_score = info["score"].pov(me).score(mate_score=MATE_SCORE)
            except Exception:
                pass

        for m in extra:
            after = clean.copy(stack=False)
            after.push(m)
            after_clean = after.copy(stack=False)
            if after_clean.is_checkmate():
                return m  # immediate mate with a card move
            if after_clean.is_game_over():
                s = 0.0
            elif after_clean.is_valid():
                try:
                    info = self._analyse(after_clean, max(1, self.depth - 1))
                    s = info["score"].pov(me).score(mate_score=MATE_SCORE)
                except Exception:
                    v = evaluate(after_clean)
                    s = v if me == chess.WHITE else -v
            else:
                v = evaluate(after_clean)
                s = v if me == chess.WHITE else -v
            if s > best_score:
                best_move, best_score = m, s

        return best_move if best_move is not None else moves[0]
