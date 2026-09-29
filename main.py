"""
Discord AI bot with automatic key/model fallback across Gemini, OpenRouter and NVIDIA.

This is main.py: bot setup, .env loading, the provider cascade, conversation
memory and the !ask command. The Tic-Tac-Toe game (!xo) lives in ttt.py and is
wired in via ttt.setup_ttt() below.

Commands
    !ask <question>                 -> tries Gemini, then OpenRouter, then NVIDIA (random personality)
    !ask gemini <question>          -> Gemini only     ("Gemini is not available" if all fail)
    !ask openrouter <question>      -> OpenRouter only ("OpenRouter is not available" if all fail)
    !ask nvidia <question>          -> NVIDIA only     ("Nvidia is not available" if all fail)
    !ask <personality> <question>   -> uses that personality (an entry in personalities.py)
    !ask clear                      -> forgets the conversation history for this channel+user

Tic-Tac-Toe (see ttt.py)
    !xo start / !xo <1-9> / !xo board / !xo quit   (aliases: !ttt, !tictactoe)
    A separate, fully code-driven game (not narrated by an LLM), so the board
    is always tracked correctly. `!xo start` shows buttons to pick a personality
    (adds flavor commentary), a difficulty (Easy / Medium / Hard) and your mark
    (X / O); typing them (`!xo start nino hard x`) skips the matching buttons.
    The personality never touches the actual board logic - it only comments.

Memory
    Each (channel, user) pair has its own rolling conversation history, so the
    bot can keep track of an ongoing chat across several !ask messages. The
    personality used also "sticks" between messages until it's changed or the
    history is reset, so a conversation stays in character. History is capped
    at MAX_HISTORY_TURNS exchanges per conversation and is saved to
    memory.json next to this script, so it survives bot restarts.

Install:  pip install -U discord.py aiohttp
Files:    main.py (this file), ttt.py (Tic-Tac-Toe), models.json (model lists),
          personalities.py (all personalities in one dictionary), _env (secrets only)
Run:      python main.py
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import random
import re
import sys
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiohttp
import discord
from discord.ext import commands

import ttt

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
PREFIX = "!"
REQUEST_TIMEOUT = 60          # seconds per API attempt
MAX_TOKENS = 2048             # max tokens for OpenRouter / NVIDIA replies
DISCORD_CHUNK = 1800          # Discord's limit is 2000 chars; keep headroom for the footer
MAX_HISTORY_TURNS = 12        # number of user+assistant exchanges kept per conversation
MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"  # conversation history storage

log = logging.getLogger("askbot")

# --------------------------------------------------------------------------- #
# Configuration
#   _env / .env      SECRETS only: Discord token + API keys. Never commit it.
#                    Real environment variables with the same names also work and
#                    win over the file (handy on hosting dashboards).
#   models.json      model lists per provider   - safe to commit, edit via PRs.
#   personalities.py one dictionary of all personalities - safe to commit, edit via PRs.
#
# Values in _env are Python literals (lists / dicts / multi-line strings), which
# python-dotenv can't parse, so they are read with ast.literal_eval instead.
# GOOGLE_MODELS / OPEN_ROUTER_MODELS / NVIDIA_MODELS / PERSONALITIES are still
# understood inside _env, but only as a fallback when models.json / personalities.py
# don't provide them, so an older setup keeps working.
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODELS_FILE = BASE_DIR / "models.json"
PERSONALITIES_FILE = BASE_DIR / "personalities.py"

ENV_KEYS = {
    "DISCORD_BOT_TOKEN": str,
    "GOOGLE_API_KEYS": list,
    "OPEN_ROUTER_API_KEYS": list,
    "NVIDIA_API_KEYS": list,
    "GOOGLE_MODELS": list,
    "OPEN_ROUTER_MODELS": list,
    "NVIDIA_MODELS": list,
    "PERSONALITIES": dict,
}
SECRET_KEYS = ("DISCORD_BOT_TOKEN", "GOOGLE_API_KEYS", "OPEN_ROUTER_API_KEYS", "NVIDIA_API_KEYS")
MODEL_FILE_KEYS = {"gemini": "GOOGLE_MODELS", "openrouter": "OPEN_ROUTER_MODELS", "nvidia": "NVIDIA_MODELS"}


def _find_env_file() -> Path:
    for name in (".env", "_env"):
        if (BASE_DIR / name).is_file():
            return BASE_DIR / name
    return BASE_DIR / ".env"


def _coerce(key: str, raw: str):
    """Turn the raw text of a setting into the type ENV_KEYS expects (None if unusable)."""
    expected = ENV_KEYS[key]
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        if expected is str:
            value = raw.strip("\"'")
        elif expected is list:
            value = raw.split(",")      # plain comma-separated text, e.g. from a hosting dashboard
        else:
            print(f"[config] Could not parse {key}: {exc}", file=sys.stderr)
            return None

    if expected is list:
        items = value if isinstance(value, (list, tuple)) else [value]
        return [str(v).strip() for v in items if str(v).strip()]
    if expected is dict:
        if not isinstance(value, dict):
            print(f"[config] {key} must be a dictionary", file=sys.stderr)
            return None
        return {str(k).strip().lower(): str(v).strip() for k, v in value.items()}
    return str(value).strip()


def load_env(path: Path) -> dict:
    """Parse KEY=<python literal> entries (values may span several lines), then let
    real environment variables override the secret ones."""
    result = {k: t() for k, t in ENV_KEYS.items()}

    if path.is_file():
        text = path.read_text(encoding="utf-8-sig")
        pattern = re.compile(r"^[ \t]*(" + "|".join(ENV_KEYS) + r")[ \t]*=[ \t]*", re.M)
        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = _coerce(m.group(1), text[m.end():end].strip())
            if value is not None:
                result[m.group(1)] = value

    for key in SECRET_KEYS:
        raw = os.environ.get(key, "").strip()
        if raw:
            value = _coerce(key, raw)
            if value:
                result[key] = value
    return result


def load_models(path: Path) -> dict:
    """models.json -> {"GOOGLE_MODELS": [...], ...}. Missing/invalid file -> {} (fallback to _env)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        print(f"[config] Could not read {path.name}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[config] {path.name} must be a JSON object like {{\"gemini\": [...]}}", file=sys.stderr)
        return {}
    result = {}
    for name, key in MODEL_FILE_KEYS.items():
        value = data.get(name)
        if isinstance(value, list):
            result[key] = [str(v).strip() for v in value if str(v).strip()]
        elif value is not None:
            print(f"[config] {path.name}: \"{name}\" must be a list of model names", file=sys.stderr)
    return result


def load_personalities(path: Path) -> dict:
    """personalities.py -> {"name": "system prompt"}.

    The file holds a plain `PERSONALITIES = {...}` dictionary of text. It is
    PARSED with ast and never executed, so a typo can't crash the bot and the
    file can't run code. Problems are printed with their line number."""
    if not path.is_file():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.name)
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"[config] Could not read {path.name}: {exc}", file=sys.stderr)
        return {}

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PERSONALITIES" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as exc:
                print(f"[config] {path.name} line {node.lineno}: PERSONALITIES may only contain "
                      f"plain text ({exc})", file=sys.stderr)
                return {}
            if not isinstance(value, dict):
                print(f"[config] {path.name}: PERSONALITIES must be a dictionary", file=sys.stderr)
                return {}
            return {str(k).strip().lower(): str(v).strip() for k, v in value.items() if str(v).strip()}

    print(f"[config] {path.name} has no `PERSONALITIES = {{...}}` dictionary", file=sys.stderr)
    return {}


CONFIG = load_env(_find_env_file())
CONFIG.update(load_models(MODELS_FILE))
_file_personalities = load_personalities(PERSONALITIES_FILE)
if _file_personalities:
    CONFIG["PERSONALITIES"] = _file_personalities
PERSONALITIES: dict[str, str] = CONFIG["PERSONALITIES"]

# --------------------------------------------------------------------------- #
# Conversation memory
# One rolling history + "sticky" personality per (channel, user), so multi-
# step interactions like a Tic-Tac-Toe game keep context and stay in
# character across several !ask messages. In-memory only (cleared on restart).
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    role: str      # "user" or "assistant"
    content: str


class MemoryStore:
    """Conversation history + sticky personality per (channel, user), persisted
    to a JSON file on disk so it survives bot restarts."""

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS, path: Path = MEMORY_FILE):
        self.max_turns = max_turns
        self.path = path
        self._lock = threading.Lock()   # guards both the in-memory dicts and the file write
        self._history: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=max_turns * 2))
        self._persona: dict[tuple[int, int], str] = {}
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
            log.warning("[memory] Could not read %s (%s) - starting with empty memory", self.path, exc)
            return

        for key_str, turns in raw.get("history", {}).items():
            try:
                key = self._key_from_str(key_str)
            except ValueError:
                continue
            dq = deque(maxlen=self.max_turns * 2)
            for t in turns[-self.max_turns * 2:]:
                dq.append(Turn(t.get("role", "user"), t.get("content", "")))
            self._history[key] = dq

        for key_str, persona in raw.get("persona", {}).items():
            try:
                key = self._key_from_str(key_str)
            except ValueError:
                continue
            self._persona[key] = persona

        log.info("[memory] Loaded %d saved conversation(s) from %s", len(self._history), self.path)

    def _save(self) -> None:
        """Write the whole store atomically (temp file + rename) so a crash
        mid-write can never corrupt memory.json."""
        data = {
            "history": {
                self._key_to_str(k): [{"role": t.role, "content": t.content} for t in v]
                for k, v in self._history.items() if v
            },
            "persona": {self._key_to_str(k): v for k, v in self._persona.items()},
        }
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("[memory] Could not save to %s: %s", self.path, exc)

    def get_history(self, ctx: commands.Context) -> list[Turn]:
        with self._lock:
            return list(self._history[self._key(ctx)])

    def add_exchange(self, ctx: commands.Context, question: str, answer: str) -> None:
        with self._lock:
            k = self._key(ctx)
            self._history[k].append(Turn("user", question))
            self._history[k].append(Turn("assistant", answer))
            self._save()

    def get_persona(self, ctx: commands.Context) -> Optional[str]:
        with self._lock:
            return self._persona.get(self._key(ctx))

    def set_persona(self, ctx: commands.Context, name: str) -> None:
        with self._lock:
            self._persona[self._key(ctx)] = name
            self._save()

    def clear(self, ctx: commands.Context) -> None:
        with self._lock:
            k = self._key(ctx)
            self._history.pop(k, None)
            self._persona.pop(k, None)
            self._save()


memory = MemoryStore()

# --------------------------------------------------------------------------- #
# Provider calls
# --------------------------------------------------------------------------- #
class ProviderError(Exception):
    """Raised when one (model, key) attempt fails.

    skip_model=True means the model itself is bad (e.g. 404), so trying the
    remaining keys for it is pointless.
    """

    def __init__(self, message: str, skip_model: bool = False, status: int = 0):
        super().__init__(message)
        self.skip_model = skip_model
        self.status = status


def mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 4 else "****"


async def post_json(session: aiohttp.ClientSession, url: str, headers: dict, payload: dict) -> dict:
    async with session.post(url, headers=headers, json=payload) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise ProviderError(f"HTTP {resp.status}: {body[:300]}",
                                skip_model=(resp.status == 404), status=resp.status)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise ProviderError(f"Invalid JSON in response: {body[:200]}")


async def call_gemini(session, api_key: str, model: str, system: str,
                      history: list["Turn"], question: str) -> str:
    model = model.removeprefix("models/")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    contents = []
    for turn in history:
        role = "user" if turn.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn.content}]})
    contents.append({"role": "user", "parts": [{"text": question}]})

    payload: dict = {"contents": contents}
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}

    data = await post_json(session, url, {"x-goog-api-key": api_key}, payload)

    candidates = data.get("candidates") or []
    parts = (candidates[0].get("content", {}).get("parts", []) if candidates else [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        reason = candidates[0].get("finishReason") if candidates else data.get("promptFeedback")
        raise ProviderError(f"Empty response ({reason})")
    return text


def strip_thinking(text: str) -> str:
    """Remove reasoning/'thinking' text so only the final answer remains."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    # Some models emit the reasoning with no opening tag, then a closing tag.
    m = list(re.finditer(r"</think(?:ing)?>", text, flags=re.I))
    if m:
        text = text[m[-1].end():]
    # Unclosed opening tag (reply cut off mid-thought) -> nothing usable.
    text = re.sub(r"<think(?:ing)?>.*\Z", "", text, flags=re.S | re.I)
    return text.strip()


async def _call_openai_compatible(session, url: str, api_key: str, model: str,
                                  system: str, history: list["Turn"], question: str,
                                  extra: Optional[dict] = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": question})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS}
    if extra:
        payload.update(extra)
    data = await post_json(session, url, headers, payload)

    # OpenRouter sometimes returns HTTP 200 with an "error" object.
    if data.get("error"):
        raise ProviderError(f"API error: {str(data['error'])[:300]}")

    choices = data.get("choices") or []
    content = (choices[0].get("message", {}).get("content") if choices else None) or ""
    content = strip_thinking(content)   # only message.content is used, never reasoning fields
    if not content:
        raise ProviderError("Empty response")
    return content


async def call_openrouter(session, api_key, model, system, history, question) -> str:
    return await _call_openai_compatible(
        session, "https://openrouter.ai/api/v1/chat/completions", api_key, model, system, history, question)


async def call_nvidia(session, api_key, model, system, history, question) -> str:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    try:
        # Ask reasoning models to skip thinking (much faster). Models that don't
        # support this option may reject it, so we retry without it.
        return await _call_openai_compatible(
            session, url, api_key, model, system, history, question,
            extra={"chat_template_kwargs": {"enable_thinking": False}})
    except ProviderError as exc:
        if exc.status not in (400, 422):
            raise
    return await _call_openai_compatible(session, url, api_key, model, system, history, question)


@dataclass
class Provider:
    name: str                       # command keyword, e.g. "gemini"
    label: str                      # display name used in messages / logs
    keys: list[str]
    models: list[str]
    call: Callable[..., Awaitable[str]]


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider("gemini", "Gemini", CONFIG["GOOGLE_API_KEYS"],
                       CONFIG["GOOGLE_MODELS"], call_gemini),
    "openrouter": Provider("openrouter", "OpenRouter", CONFIG["OPEN_ROUTER_API_KEYS"],
                           CONFIG["OPEN_ROUTER_MODELS"], call_openrouter),
    "nvidia": Provider("nvidia", "Nvidia", CONFIG["NVIDIA_API_KEYS"],
                       CONFIG["NVIDIA_MODELS"], call_nvidia),
}
PROVIDER_ORDER = ["gemini", "openrouter", "nvidia"]   # order used by plain !ask / !ask <personality>


async def run_provider(session, prov: Provider, system: str, history: list["Turn"],
                       question: str) -> Optional[tuple[str, str]]:
    """Try every model x key of one provider. Returns (text, model) on first success."""
    if not prov.keys or not prov.models:
        log.warning("[%s] No keys or models configured - skipping", prov.label)
        return None

    for model in prov.models:
        for key in prov.keys:
            log.info("[%s] Trying model=%s key=%s", prov.label, model, mask(key))
            try:
                text = await asyncio.wait_for(
                    prov.call(session, key, model, system, history, question), timeout=REQUEST_TIMEOUT + 5)
            except ProviderError as exc:
                log.warning("[%s] FAILED model=%s key=%s -> %s", prov.label, model, mask(key), exc)
                if exc.skip_model:
                    log.warning("[%s] Skipping remaining keys for model=%s", prov.label, model)
                    break
            except Exception as exc:  # network errors, timeouts, etc.
                log.warning("[%s] ERROR model=%s key=%s -> %s: %s",
                            prov.label, model, mask(key), type(exc).__name__, exc)
            else:
                log.info("[%s] SUCCESS model=%s key=%s", prov.label, model, mask(key))
                return text, model
    log.error("[%s] All models/keys failed", prov.label)
    return None


async def run_cascade(session, providers: list[Provider], system: str,
                      history: list["Turn"], question: str):
    for prov in providers:
        result = await run_provider(session, prov, system, history, question)
        if result:
            return result
    return None


# --------------------------------------------------------------------------- #
# Discord bot
# --------------------------------------------------------------------------- #
def split_message(text: str, limit: int = DISCORD_CHUNK) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    chunks.append(text)
    return chunks


def build_system_prompt(personality: str, user_name: str) -> str:
    """Personality text + info about who is asking, so answers are for that specific user."""
    user_info = (
        f"\n\n[Context about the person you are talking to]\n"
        f"Their Discord username is \"{user_name}\". You are replying directly to them. "
        f"If they ask about their name or who they are, answer with this username "
        f"(e.g. \"Your name is {user_name}\", phrased in your own style). "
        f"Use this username when addressing them, unless the personality instructions "
        f"above explicitly define a different name for the user (for example a husband's "
        f"name), in which case follow the personality instructions."
    )
    return (personality + user_info).strip()


class AskBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # must also be enabled in the Developer Portal
        super().__init__(
            command_prefix=PREFIX,
            intents=intents,
            help_command=None,
            allowed_mentions=discord.AllowedMentions.none(),  # model output can't ping anyone
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def setup_hook(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT))

    async def close(self):
        if self.session:
            await self.session.close()
        await super().close()

    async def on_ready(self):
        log.info("Logged in as %s (id=%s)", self.user, self.user.id)
        log.info("Personalities loaded: %s", ", ".join(PERSONALITIES) or "none")
        for p in PROVIDERS.values():
            log.info("%s: %d key(s), %d model(s)", p.label, len(p.keys), len(p.models))

    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            return
        log.exception("Command error", exc_info=error)


bot = AskBot()

ttt.setup_ttt(
    bot,
    personalities=PERSONALITIES,
    providers=PROVIDERS,
    provider_order=PROVIDER_ORDER,
    build_system_prompt=build_system_prompt,
    run_cascade=run_cascade,
    prefix=PREFIX,
)


CLEAR_WORDS = {"clear", "reset", "forget", "new"}


@bot.command(name="ask")
async def ask(ctx: commands.Context, *, args: str = ""):
    parts = args.split(maxsplit=1)
    if not parts:
        await ctx.reply(
            f"Usage: `{PREFIX}ask <question>`, `{PREFIX}ask gemini|openrouter|nvidia <question>`, "
            f"`{PREFIX}ask <personality> <question>` or `{PREFIX}ask clear`", mention_author=False)
        return

    first = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    # !ask clear / reset / forget / new -> wipe this channel+user's conversation
    if first in CLEAR_WORDS and not rest:
        memory.clear(ctx)
        await ctx.reply("Conversation memory cleared - starting fresh.", mention_author=False)
        return

    provider_only: Optional[Provider] = None
    persona_name: Optional[str] = None
    explicit_persona = False

    if first in PROVIDERS:
        provider_only, question = PROVIDERS[first], rest
    elif first in PERSONALITIES:
        persona_name, question = first, rest
        explicit_persona = True
    else:
        question = args.strip()

    if not question:
        await ctx.reply("Please add a question after the command.", mention_author=False)
        return

    # Explicitly naming a personality different from the current one starts a
    # fresh conversation, so an old game/context doesn't bleed into the new persona.
    if explicit_persona and memory.get_persona(ctx) not in (None, persona_name):
        memory.clear(ctx)
        log.info("!ask from %s | personality switched to %s -> memory cleared", ctx.author, persona_name)
        await ctx.send(f"\U0001F504 Switching to **{persona_name}** - starting a fresh conversation.")

    # If no personality was explicitly named this message, stick with whatever
    # personality was last used in this channel+user's conversation (so an
    # ongoing game/chat stays in character), falling back to random.
    if persona_name is None:
        persona_name = memory.get_persona(ctx)
        if persona_name not in PERSONALITIES:
            persona_name = random.choice(list(PERSONALITIES)) if PERSONALITIES else ""

    user_name = ctx.author.display_name  # use ctx.author.name for the raw @username instead
    system = build_system_prompt(PERSONALITIES.get(persona_name, ""), user_name)
    history = memory.get_history(ctx)

    providers = [provider_only] if provider_only else [PROVIDERS[n] for n in PROVIDER_ORDER]
    log.info("!ask from %s | personality=%s | scope=%s | history=%d turn(s) | question=%r",
             ctx.author, persona_name or "none",
             provider_only.label if provider_only else "all providers", len(history) // 2, question[:200])

    async with ctx.typing():
        result = await run_cascade(bot.session, providers, system, history, question)

    if result is None:
        msg = f"{provider_only.label} is not available" if provider_only else "No AI provider is available right now."
        await ctx.reply(msg, mention_author=False)
        return

    text, model = result
    memory.add_exchange(ctx, question, text)
    if persona_name:
        memory.set_persona(ctx, persona_name)

    chunks = split_message(text)
    chunks[-1] += f"\n\n**Model:** `{model}`"
    await ctx.reply(chunks[0], mention_author=False)
    for chunk in chunks[1:]:
        await ctx.send(chunk)


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("discord").setLevel(logging.WARNING)

    token = CONFIG["DISCORD_BOT_TOKEN"]
    if not token or token.startswith("#"):
        log.critical("DISCORD_BOT_TOKEN is missing in %s", _find_env_file())
        sys.exit(1)
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
