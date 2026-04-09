# Link Manager Bot

Telegram bot for saving, organizing and searching links with AI-powered semantic search.

## Overview

This bot was built as a hackathon project. It allows users to save links with descriptions, organize them into folders, share folders with others, and search through saved links using natural language.

The search uses an LLM (Qwen) to find relevant links by meaning, not just by keyword matching. The bot can understand requests like "find python tutorials" or "show my work links".

## Tech Stack

- Python 3.12 + aiogram 3.x (Telegram bot framework)
- PostgreSQL (asyncpg) for data storage
- Qwen LLM via local proxy (OpenAI-compatible API)
- Docker Compose for deployment

The architecture is similar to what we built in Labs 7 and 8 — a Telegram bot that uses LLM for intent recognition and semantic search, with PostgreSQL for persistence.

## Features

### Basic Commands

- `/add <url> "<description>"` — save a link
- `/find <query>` — search saved links using AI
- `/list` — show all saved links
- `/folders` — list all folders
- `/create <name>` — create a new folder
- `/folder <name>` — open a folder and show its links
- `/delete <url or id>` — delete a link

### Advanced Commands

- `/share <name>` — generate a share key for a folder
- `/join <key>` — join a shared folder
- `/revoke <name>` — revoke access (owner only)
- `/share_list <name>` — show who has access
- `/edit <url> "<new description>"` — edit link description
- `/delete_folder <name>` — delete an empty folder

### Natural Language

The bot also understands plain text input (experimental):
- `find python tutorial` — searches links
- `add https://python.org docs` — saves a link
- `show folders` — lists folders
- `create folder Work` — creates a folder

## How LLM Integration Works

When a user sends a message that is not a command, the bot:
1. Sends the text to Qwen (via local proxy at localhost:42005)
2. Qwen returns a JSON with intent ("search", "add", "list_folders", etc.) and extracted entities
3. The bot executes the corresponding action

For `/find` command, the bot collects all user's link descriptions, sends them to Qwen with the search query, and Qwen returns IDs of the 3 most relevant links. Results are cached for 1 hour.

## How Sharing Works

- Owner runs `/share <folder>` → bot generates a unique key like "Work_abc123"
- Other users run `/join <key>` → they get read or write access to that folder
- Shared folders appear in `/folders` and `/list` with a 🔗 icon
- With write access, users can add their own links to the shared folder
- Owner can revoke access with `/revoke <folder>` — all guests lose access immediately

## How Delete Works

- `/delete <url>` — deletes ALL links with that URL (from user's own folders and shared folders where user has write access)
- `/delete <id>` — deletes a single link by its database ID
- Cannot delete links from System folder or from shared folders where user has only read access
- `/delete_folder <name>` — deletes an empty folder (user's own folders only, not shared or System)

## Limitations

- LLM runs on CPU, so search requests take 9-15 seconds. Use `/commands` for instant responses.
- The Qwen proxy requires an OAuth token that expires every 6 hours. The token must be refreshed manually or via a helper script.
- Natural language understanding works for simple phrases but may fail on complex sentences. Use commands for precise operations.
- Code contains commented sections and debug logs that need cleanup.
