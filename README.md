# Discord AI bot

Chat bot with automatic fallback across Gemini, OpenRouter and NVIDIA, plus a
Tic-Tac-Toe game (`!xo`) with personalities and Easy / Medium / Hard difficulty.

## Files

| File | What it is | Commit it? |
|---|---|---|
| `main.py`, `ttt.py` | the bot | yes |
| `models.json` | model names to try, per provider (tried top to bottom) | yes |
| `personalities.py` | all personalities in one dictionary (add a new entry to add a personality) | yes |
| `_env.example` | template for your secrets | yes |
| `_env` | **your real token and API keys** | **never** (it is in `.gitignore`) |
| `memory.json`, `games.json` | per-user chat history and game state, created at runtime | no |

## Run it

1. `pip install -U discord.py aiohttp`
2. Copy `_env.example` to `_env` and fill in your own keys.
3. `python main.py`

To test, create **your own** bot in the Discord Developer Portal and your own free
API keys. Don't ask for anyone else's token. Never paste keys into a chat, issue or pull request.

## Contributing

- **Add or edit a personality:** add or change an entry in `personalities.py`
  (`"name": """system prompt text""",` - the name is one lowercase word).
  It shows up as a button in `!xo start` and works with `!ask <name> ...`.
  The file is only read, never run, and a typo is reported with its line number.
- **Change models:** edit `models.json`. Check each name exists on the provider.
- Open a pull request. No secrets are involved in either of the two.
