"""
Tic-Tac-Toe game, split out from main.py, exposed to Discord as the !xo command.

This is a real, code-driven game engine (board state + win detection + a
minimax opponent) rather than asking an LLM to track a board in text, which is
unreliable. One game per (channel, user), persisted to disk.

Interactive setup: `!xo start` shows clickable buttons for the personality and
for your mark (X / O). Only the person who ran the command can click them.

Difficulty (Easy / Medium / Hard, chosen in `!xo start`): each difficulty is the
chance per move that the bot plays its best move (otherwise a random move), see
DIFFICULTIES below. Hard is perfect play, so the best you can do is draw.

An optional personality (from main.py's PERSONALITIES) adds a short in-character
line. The game result and that line are sent together in ONE message; if the
AI is slow (> FLAVOR_WAIT s) the game reply goes out first and the line is added
by editing that same message when it arrives. The personality never touches the
board logic, so it can't break the game.

Usage from main.py (unchanged):

    import ttt
    ...
    ttt.setup_ttt(
        bot,
        personalities=PERSONALITIES,
        providers=PROVIDERS,
        provider_order=PROVIDER_ORDER,
        build_system_prompt=build_system_prompt,
        run_cascade=run_cascade,
        prefix=PREFIX,
    )

setup_ttt() registers the !xo command (aliases: !ttt, !tictactoe).
Requires discord.py 2.x (buttons / views).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import discord
from discord.ext import commands

log = logging.getLogger("askbot.ttt")

GAMES_FILE = Path(__file__).resolve().parent / "games.json"

# --- Tunables ---------------------------------------------------------------
DIFFICULTIES = {        # chance per move that the bot plays its best move (else a random cell)
    "easy": 0.30,       # a sensible player wins most games
    "medium": 0.65,     # a real contest
    "hard": 1.00,       # perfect play - cannot be beaten, only drawn
}
DEFAULT_DIFFICULTY = "medium"
DIFFICULTY_ALIASES = {"med": "medium", "normal": "medium"}
SETUP_TIMEOUT = 60      # seconds the person has to click a setup button
FLAVOR_WAIT = 20        # seconds the game reply is held back waiting for the personality line
FLAVOR_MAX = 90         # if the line isn't ready in FLAVOR_WAIT, keep trying this long in the
                        # background and EDIT it into the already-sent message when it arrives
NONE_VALUE = "__none__" # button value for "no personality"
MAX_PERSONALITY_BUTTONS = 24   # Discord allows 25 buttons per message; 1 is "No personality"

# --------------------------------------------------------------------------- #
# Game engine (board, win detection, unbeatable opponent)
# --------------------------------------------------------------------------- #
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def ttt_winner(board: list[str]) -> Optional[str]:
    """Returns 'X', 'O', 'draw', or None (game still in progress)."""
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(board):
        return "draw"
    return None


@lru_cache(maxsize=None)
def _ttt_score(board: tuple, current_mark: str, bot_mark: str) -> int:
    """Perfect-play value of `board` (cached, so the whole game tree is only
    ever computed once). Positive = good for the bot. Faster wins score higher
    and faster losses score lower, so the bot takes an immediate win instead of
    dawdling."""
    winner = ttt_winner(list(board))
    empties = board.count("")
    if winner == bot_mark:
        return 10 + empties
    if winner == "draw":
        return 0
    if winner is not None:
        return -(10 + empties)

    other_mark = "O" if current_mark == "X" else "X"
    scores = [
        _ttt_score(board[:i] + (current_mark,) + board[i + 1:], other_mark, bot_mark)
        for i, v in enumerate(board) if not v
    ]
    return max(scores) if current_mark == bot_mark else min(scores)


def ttt_best_moves(board: list[str], bot_mark: str) -> list[int]:
    """All equally-best cells for the bot (perfect play)."""
    b = tuple(board)
    other_mark = "O" if bot_mark == "X" else "X"
    scored = [
        (_ttt_score(b[:i] + (bot_mark,) + b[i + 1:], other_mark, bot_mark), i)
        for i, v in enumerate(b) if not v
    ]
    top = max(s for s, _ in scored)
    return [i for s, i in scored if s == top]


def ttt_bot_move(board: list[str], bot_mark: str, difficulty: str = DEFAULT_DIFFICULTY) -> int:
    """Best move with the probability set by `difficulty` (see DIFFICULTIES),
    otherwise a random empty cell. Ties between equally good moves are broken
    randomly, so the bot doesn't open the same way every game."""
    if random.random() < DIFFICULTIES.get(difficulty, DIFFICULTIES[DEFAULT_DIFFICULTY]):
        return random.choice(ttt_best_moves(board, bot_mark))
    return random.choice([i for i, v in enumerate(board) if not v])


def ttt_result_text(winner: str, human_mark: str) -> str:
    if winner == "draw":
        return "It's a draw! \U0001F91D"
    return "You win! \U0001F389" if winner == human_mark else "I win! \U0001F916"


def render_ttt_board(board: list[str]) -> str:
    cells = [board[i] if board[i] else str(i + 1) for i in range(9)]
    rows = [" | ".join(f"{c:^1}" for c in cells[r:r + 3]) for r in (0, 3, 6)]
    return "```\n" + "\n---+---+---\n".join(f" {row} " for row in rows) + "\n```"


def parse_personality_choice(text: str, options: list[str]) -> Optional[str]:
    """Match a user's reply to a personality by number or by name."""
    t = text.strip().lower()
    if not t:
        return None
    if t.isdigit():
        idx = int(t) - 1
        return options[idx] if 0 <= idx < len(options) else None
    return t if t in options else None


def parse_mark_choice(text: str) -> Optional[str]:
    """Match a user's reply to 'X' or 'O'. Returns None if it's neither."""
    t = text.strip().lower()
    if t == "x":
        return "X"
    if t == "o":
        return "O"
    return None


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
@dataclass
class TTTGame:
    board: list[str] = field(default_factory=lambda: [""] * 9)
    finished: bool = False
    personality: Optional[str] = None   # which personality flavors this game's commentary
    human_mark: str = "X"                # which mark the person is playing
    bot_mark: str = "O"                  # which mark the bot is playing (always the other one)
    difficulty: str = DEFAULT_DIFFICULTY # "easy" | "medium" | "hard"


class GameStore:
    """Persists one Tic-Tac-Toe game per (channel, user) to disk, atomically
    (temp file + rename) so a crash mid-write can't corrupt games.json."""

    def __init__(self, path: Path = GAMES_FILE):
        self.path = path
        self._lock = threading.Lock()
        self._games: dict[tuple[int, int], TTTGame] = {}
        self._load()

    @staticmethod
    def _key(ctx: commands.Context) -> tuple[int, int]:
        return (ctx.channel.id, ctx.author.id)

    @staticmethod
    def _key_to_str(key: tuple[int, int]) -> str:
        return f"{key[0]}:{key[1]}"

    @staticmethod
    def _key_from_str(s: str) -> tuple[int, int]:
        channel_id, user_id = s.split(":")
        return (int(channel_id), int(user_id))

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read %s (%s) - starting with no saved games", self.path, exc)
            return
        for key_str, g in raw.items():
            try:
                key = self._key_from_str(key_str)
            except ValueError:
                continue
            board = g.get("board", [""] * 9)
            if not isinstance(board, list) or len(board) != 9:
                continue
            self._games[key] = TTTGame(board=board, finished=bool(g.get("finished", False)),
                                       personality=g.get("personality"),
                                       human_mark=g.get("human_mark", "X"),
                                       bot_mark=g.get("bot_mark", "O"),
                                       difficulty=g.get("difficulty") if g.get("difficulty") in DIFFICULTIES
                                       else DEFAULT_DIFFICULTY)
        log.info("Loaded %d saved Tic-Tac-Toe game(s) from %s", len(self._games), self.path)

    def _save(self) -> None:
        data = {
            self._key_to_str(k): {"board": g.board, "finished": g.finished, "personality": g.personality,
                                  "human_mark": g.human_mark, "bot_mark": g.bot_mark,
                                  "difficulty": g.difficulty}
            for k, g in self._games.items()
        }
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("Could not save to %s: %s", self.path, exc)

    def get(self, ctx: commands.Context) -> Optional[TTTGame]:
        with self._lock:
            return self._games.get(self._key(ctx))

    def start(self, ctx: commands.Context) -> TTTGame:
        with self._lock:
            game = TTTGame()
            self._games[self._key(ctx)] = game
            self._save()
            return game

    def save(self, ctx: commands.Context, game: TTTGame) -> None:
        with self._lock:
            self._games[self._key(ctx)] = game
            self._save()

    def end(self, ctx: commands.Context) -> bool:
        with self._lock:
            existed = self._games.pop(self._key(ctx), None) is not None
            if existed:
                self._save()
            return existed


games = GameStore()


# --------------------------------------------------------------------------- #
# Discord command wiring
# --------------------------------------------------------------------------- #
def setup_ttt(
    bot: commands.Bot,
    *,
    personalities: dict[str, str],
    providers: dict,
    provider_order: list[str],
    build_system_prompt: Callable[[str, str], str],
    run_cascade: Callable,
    prefix: str = "!",
) -> None:
    """Registers the !xo Tic-Tac-Toe command on `bot`. Everything the command
    needs from main.py (personalities, providers, prompt helpers) is passed in
    explicitly rather than imported, so this module has no circular import on
    main.py."""

    TTT_HELP = (
        "**Tic-Tac-Toe** - pick a personality, a difficulty and your mark.\n"
        f"`{prefix}xo start` - start a new game; buttons ask for personality, difficulty (Easy/Medium/Hard) and mark (X/O)\n"
        f"`{prefix}xo start <personality> <easy|medium|hard> <x|o>` - skip the buttons you already know "
        f"(any order, any of them optional), e.g. `{prefix}xo start hard x`\n"
        f"`{prefix}xo start none` - start a plain game with no personality flavor\n"
        f"`{prefix}xo <1-9>` - place your mark in that cell\n"
        f"`{prefix}xo board` - show the current board\n"
        f"`{prefix}xo quit` - end the game\n"
        "```\n 1 | 2 | 3 \n---+---+---\n 4 | 5 | 6 \n---+---+---\n 7 | 8 | 9 \n```\n"
        "X always goes first - if you pick O, the bot takes the first move."
    )

    class ChoiceView(discord.ui.View):
        """A row of buttons only `owner_id` may click. After view.wait()
        returns, `choice` is the clicked value (None = timed out) and
        `interaction` is the click, still un-responded so the caller can
        edit the message with the next step."""

        def __init__(self, owner_id: int, options: list[tuple[str, str, discord.ButtonStyle]]):
            super().__init__(timeout=SETUP_TIMEOUT)
            self.owner_id = owner_id
            self.choice: Optional[str] = None
            self.interaction: Optional[discord.Interaction] = None
            self.message: Optional[discord.Message] = None
            for label, value, style in options:
                button = discord.ui.Button(label=label[:80], style=style)
                button.callback = self._make_callback(value)
                self.add_item(button)

        def _make_callback(self, value: str):
            async def callback(interaction: discord.Interaction):
                if self.choice is not None:          # double click
                    await interaction.response.defer()
                    return
                self.choice = value
                self.interaction = interaction
                self.stop()
            return callback

        async def interaction_check(self, interaction: discord.Interaction) -> bool:
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message(
                    f"These buttons aren't yours - start your own game with `{prefix}xo start`.",
                    ephemeral=True)
                return False
            return True

    async def ask_choice(ctx: commands.Context, prompt: str,
                         options: list[tuple[str, str, discord.ButtonStyle]],
                         prev: Optional[discord.Interaction]):
        """Show `prompt` with buttons and wait for a click. The first step
        sends a new reply; later steps edit that same message in place.
        Returns (value, interaction), or None if it timed out."""
        view = ChoiceView(ctx.author.id, options)
        if prev is not None:
            await prev.response.edit_message(content=prompt, view=view)
            view.message = prev.message
        else:
            view.message = await ctx.reply(prompt, view=view, mention_author=False)
        await view.wait()
        if view.choice is None:
            try:
                await view.message.edit(
                    content=f"\u231B Timed out - no game started. Use `{prefix}xo start` to try again.",
                    view=None)
            except discord.HTTPException:
                pass
            return None
        return view.choice, view.interaction

    async def ttt_flavor_line(persona_name: str, user_name: str, situation: str) -> Optional[str]:
        """Best-effort short in-character reaction to a game event. The game
        itself never depends on this succeeding. Never raises; returns None
        if no provider answered within FLAVOR_MAX seconds."""
        system = build_system_prompt(personalities.get(persona_name, ""), user_name)
        system += (
            "\n\nYou are reacting to a real Tic-Tac-Toe game that is tracked and played "
            "correctly elsewhere - you are NOT playing it or tracking the board yourself. "
            "Reply with ONE short in-character sentence reacting to what just happened. "
            "Do not describe the full board, do not invent or suggest moves, no more than one sentence."
        )
        try:
            result = await asyncio.wait_for(
                run_cascade(bot.session, [providers[n] for n in provider_order], system, [], situation),
                timeout=FLAVOR_MAX)
        except asyncio.TimeoutError:
            log.warning("Flavor commentary: no provider answered within %ss - no personality line", FLAVOR_MAX)
            return None
        except Exception:
            log.exception("Flavor commentary call raised unexpectedly")
            return None
        if not result:
            log.warning("Flavor commentary: every provider/model/key failed - no personality line "
                        "(see the [Gemini]/[OpenRouter]/[Nvidia] FAILED lines above)")
            return None
        log.info("Flavor commentary OK (model=%s)", result[1])
        lines = result[0].strip().splitlines()
        return lines[0][:300] if lines else None

    background: set = set()   # keeps late-edit tasks alive until they finish

    def start_flavor(ctx: commands.Context, persona_name: Optional[str], situation: str):
        """Begin fetching the personality line in the background (None if no personality)."""
        if not persona_name:
            return None
        return asyncio.ensure_future(ttt_flavor_line(persona_name, ctx.author.display_name, situation))

    async def wait_flavor(ctx: commands.Context, task) -> Optional[str]:
        """Hold the reply for up to FLAVOR_WAIT seconds. Returns the line if it
        is ready in time, else None (the task keeps running in the background)."""
        if task is None:
            return None
        async with ctx.typing():
            await asyncio.wait({task}, timeout=FLAVOR_WAIT)
        return task.result() if task.done() else None

    def add_late_flavor(task, text: str, edit: Callable) -> None:
        """The line wasn't ready when the reply was sent: when it arrives, edit
        it into that same message via `edit(new_content)`."""
        if task is None:
            return

        async def runner():
            try:
                line = await task
                if line:
                    await edit(with_flavor(text, line))
            except Exception:
                log.exception("Could not add late personality line to the game message")

        t = asyncio.ensure_future(runner())
        background.add(t)
        t.add_done_callback(background.discard)

    async def send_reply(ctx: commands.Context, text: str, persona_name: Optional[str], situation: str) -> None:
        """Send `text` as one message with the personality's reaction to
        `situation` (same wait / late-edit behaviour as a normal move). With no
        personality it is just a plain reply."""
        task = start_flavor(ctx, persona_name, situation)
        line = await wait_flavor(ctx, task)
        sent = await ctx.reply(with_flavor(text, line), mention_author=False)
        if line is None:
            add_late_flavor(task, text, lambda c: sent.edit(content=c))

    def with_flavor(text: str, line: Optional[str]) -> str:
        """Game text and personality line in ONE message (line as a quote)."""
        return f"{text}\n> {line}" if line else text

    @bot.command(name="xo", aliases=["ttt", "tictactoe"])
    async def xo(ctx: commands.Context, *, args: str = ""):
        """A real (non-AI-narrated) Tic-Tac-Toe game, so the board is always
        tracked correctly instead of relying on a model to remember it in text.
        An optional personality can add flavor commentary on top of each move."""
        raw = args.strip()
        tokens = raw.split(maxsplit=1)
        first = tokens[0].lower() if tokens else ""

        if first in ("", "help"):
            await ctx.reply(TTT_HELP, mention_author=False)
            return

        if first in ("quit", "end", "stop"):
            if games.end(ctx):
                await ctx.reply("Game ended.", mention_author=False)
            else:
                await ctx.reply("You don't have a game in progress.", mention_author=False)
            return

        if first == "start":
            rest = tokens[1].strip() if len(tokens) > 1 else ""

            mark_token = ""
            difficulty_token = ""
            leftover: list[str] = []
            for tok in rest.split():
                low = tok.lower()
                low_diff = DIFFICULTY_ALIASES.get(low, low)
                if low in ("x", "o") and not mark_token:
                    mark_token = low.upper()
                elif low_diff in DIFFICULTIES and not difficulty_token and low not in personalities:
                    difficulty_token = low_diff
                else:
                    leftover.append(tok)
            persona_choice = leftover[0].lower() if leftover else ""

            notes = ""
            persona_name: Optional[str] = None
            need_persona = False
            if persona_choice in ("none", "no", "off"):
                pass
            elif persona_choice:
                if persona_choice in personalities:
                    persona_name = persona_choice
                else:
                    notes = (f"\n\u26A0\uFE0F Unknown personality '{persona_choice}' - started without one. "
                             f"(Available: {', '.join(personalities) or 'none configured'})")
            else:
                need_persona = bool(personalities)

            prev: Optional[discord.Interaction] = None

            # Step 1: personality buttons
            if need_persona:
                names = list(personalities)[:MAX_PERSONALITY_BUTTONS]
                options = [(n.capitalize(), n, discord.ButtonStyle.primary) for n in names]
                options.append(("No personality", NONE_VALUE, discord.ButtonStyle.secondary))
                picked = await ask_choice(
                    ctx, "\U0001F3AD **Choose a personality** to comment on your game:", options, prev)
                if picked is None:
                    return
                value, prev = picked
                persona_name = None if value == NONE_VALUE else value

            # Step 2: difficulty buttons
            if difficulty_token:
                difficulty = difficulty_token
            else:
                options = [("\U0001F7E2  Easy", "easy", discord.ButtonStyle.success),
                           ("\U0001F7E1  Medium", "medium", discord.ButtonStyle.primary),
                           ("\U0001F534  Hard", "hard", discord.ButtonStyle.danger)]
                picked = await ask_choice(
                    ctx, "\U0001F3AF **Choose a difficulty:**\n"
                         "\U0001F7E2 **Easy** - I make plenty of mistakes\n"
                         "\U0001F7E1 **Medium** - a real contest\n"
                         "\U0001F534 **Hard** - I play perfectly, the best you can do is a draw",
                    options, prev)
                if picked is None:
                    return
                difficulty, prev = picked

            # Step 3: X / O buttons
            if mark_token:
                human_mark = mark_token
            else:
                options = [("\u274C  Play as X (go first)", "X", discord.ButtonStyle.primary),
                           ("\u2B55  Play as O", "O", discord.ButtonStyle.success)]
                picked = await ask_choice(ctx, "Which mark do you want? **X** always goes first.", options, prev)
                if picked is None:
                    return
                human_mark, prev = picked
            bot_mark = "O" if human_mark == "X" else "X"

            # Create the game (only now, so cancelling/timing out never wipes a game in progress)
            game = games.start(ctx)
            game.personality = persona_name
            game.human_mark = human_mark
            game.bot_mark = bot_mark
            game.difficulty = difficulty

            first_move_note = ""
            opening_cell = -1
            if bot_mark == "X":   # X always goes first, so the bot opens if the person picked O
                opening_cell = ttt_bot_move(game.board, bot_mark, difficulty)
                game.board[opening_cell] = bot_mark
                first_move_note = f" I go first as **X** and I'll take **{opening_cell + 1}**."
            games.save(ctx, game)

            if prev is not None:   # acknowledge the click right away; final text replaces this
                await prev.response.edit_message(content="\u23F3 Setting up your game...", view=None)

            situation = "A new Tic-Tac-Toe game against your opponent just began."
            if first_move_note:
                situation += f" You went first (as {bot_mark}) and took cell {opening_cell + 1}."
            flavor_task = start_flavor(ctx, persona_name, situation)
            line = await wait_flavor(ctx, flavor_task)

            persona_note = f" I'll play with some **{persona_name}** flavor." if persona_name else ""
            base = (f"New game started! You're **{human_mark}**, I'm **{bot_mark}**. "
                    f"Difficulty: **{difficulty.capitalize()}**.{first_move_note} "
                    f"Play a cell with `{prefix}xo <1-9>`.{persona_note}{notes}\n{render_ttt_board(game.board)}")
            content = with_flavor(base, line)
            edit = None
            if prev is not None:
                try:
                    await prev.edit_original_response(content=content, view=None)
                    edit = lambda c: prev.edit_original_response(content=c, view=None)
                except discord.HTTPException:
                    pass
            if edit is None:
                sent = await ctx.reply(content, mention_author=False)
                edit = lambda c: sent.edit(content=c)
            if line is None:
                add_late_flavor(flavor_task, base, edit)
            return

        if first == "board":
            game = games.get(ctx)
            if not game:
                await ctx.reply(f"No game in progress. Start one with `{prefix}xo start`.", mention_author=False)
                return
            await ctx.reply(render_ttt_board(game.board), mention_author=False)
            return

        if not (first.isascii() and first.isdigit() and 1 <= int(first) <= 9):
            existing = games.get(ctx)   # a finished game still remembers its personality
            await send_reply(
                ctx, f"Not sure what you mean. Try `{prefix}xo help`.",
                existing.personality if existing else None,
                f"The player typed an invalid Tic-Tac-Toe move: '{raw[:50]}'. "
                f"Valid moves are the cell numbers 1 to 9.")
            return

        game = games.get(ctx)
        if not game:
            await ctx.reply(f"No game in progress. Start one with `{prefix}xo start`.", mention_author=False)
            return
        if game.finished:
            await send_reply(
                ctx, f"That game is already over. Start a new one with `{prefix}xo start`.",
                game.personality,
                "The player tried to make a move, but the Tic-Tac-Toe game is already over.")
            return

        cell = int(first) - 1
        if game.board[cell]:
            await send_reply(
                ctx, f"Cell {first} is already taken.\n{render_ttt_board(game.board)}",
                game.personality,
                f"The player tried to play cell {first}, but that cell is already taken. "
                f"They have to pick a free cell.")
            return

        # Human move
        game.board[cell] = game.human_mark
        winner = ttt_winner(game.board)
        bot_cell: Optional[int] = None

        if not winner:
            # Bot's move (strength depends on the game's difficulty - see DIFFICULTIES)
            bot_cell = ttt_bot_move(game.board, game.bot_mark, game.difficulty)
            game.board[bot_cell] = game.bot_mark
            winner = ttt_winner(game.board)

        game.finished = bool(winner)
        games.save(ctx, game)   # state is saved BEFORE any slow commentary call

        # What happened, for the personality to react to
        if bot_cell is None:
            outcome = "won them the game" if winner == game.human_mark else "ended the game in a draw"
            situation = f"The player just placed {game.human_mark} in cell {first}, which {outcome}."
            text = f"{render_ttt_board(game.board)}\n{ttt_result_text(winner, game.human_mark)}"
        else:
            if winner == game.bot_mark:
                situation = f"The player played cell {first}, then you played cell {bot_cell + 1} and won the game."
            elif winner == "draw":
                situation = f"The player played cell {first}, then you played cell {bot_cell + 1}, ending in a draw."
            else:
                situation = f"The player played cell {first}, then you played cell {bot_cell + 1}. The game continues."
            text = f"You played **{first}**, I'll take **{bot_cell + 1}**.\n{render_ttt_board(game.board)}"
            if winner:
                text += f"\n{ttt_result_text(winner, game.human_mark)}"

        flavor_task = start_flavor(ctx, game.personality, situation)
        line = await wait_flavor(ctx, flavor_task)

        # One message: the game output together with the personality's line.
        # If the AI is slow, the line is edited into this same message when ready.
        sent = await ctx.reply(with_flavor(text, line), mention_author=False)
        if line is None:
            add_late_flavor(flavor_task, text, lambda c: sent.edit(content=c))
