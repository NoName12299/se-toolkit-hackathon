#!/usr/bin/env python3
"""
Telegram bot for saving and searching links via LLM-powered semantic search.

Commands:
    /add <url> "description" [--folder <name>]  — save a link with description
    /find <query> [--folder <name>]             — find relevant links using LLM
    /folders                                    — list your folders
    /create <name>                              — create a folder
    /folder <name>                              — show links in a folder
    /delete <id>                                — delete a link
    /list                                       — show all your links

Architecture:
    - aiogram 3.x for Telegram bot framework
    - asyncpg for async PostgreSQL connection pool
    - httpx for async HTTP calls to LLM API
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import suppress

import asyncpg
import httpx
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

# ---------------------------------------------------------------------------
# Configuration — loaded from environment variables
# ---------------------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "my-secret-qwen-key")
LLM_MODEL = os.getenv("LLM_MODEL", "coder-model")

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "linksdb")
DB_USER = os.getenv("DB_USER", "linksuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "linkspassword")

# Cache TTL in seconds
CACHE_TTL = 3600  # 1 hour

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

DB_POOL: asyncpg.Pool | None = None


async def init_db_pool() -> asyncpg.Pool:
    """Create and return an asyncpg connection pool."""
    dsn = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    logger.info("Connecting to database: %s", dsn.replace(DB_PASSWORD, "****"))
    pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)
    return pool


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create tables and indexes if they don't exist, and migrate existing tables."""
    async with pool.acquire() as conn:
        # Create folders table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id          SERIAL PRIMARY KEY,
                user_id     BIGINT    NOT NULL,
                name        TEXT      NOT NULL,
                is_system   BOOLEAN   DEFAULT FALSE,
                created_at  TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_folders_user_id ON folders(user_id);
        """)

        # Create links table if not exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS links (
                id          SERIAL PRIMARY KEY,
                user_id     BIGINT    NOT NULL,
                url         TEXT      NOT NULL,
                description TEXT      NOT NULL,
                folder_id   INT       REFERENCES folders(id) ON DELETE SET NULL,
                created_at  TIMESTAMP DEFAULT NOW()
            );
        """)

        # Add folder_id column if it doesn't exist (migration for existing tables)
        await conn.execute("""
            ALTER TABLE links ADD COLUMN IF NOT EXISTS folder_id INT REFERENCES folders(id) ON DELETE SET NULL;
        """)

        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_user_id ON links(user_id);
            CREATE INDEX IF NOT EXISTS idx_links_folder_id ON links(folder_id);
        """)

        logger.info("Schema ensured (folders and links tables exist, folder_id column present)")


# --- Folder DB helpers ---

async def create_folder(pool: asyncpg.Pool, user_id: int, name: str, is_system: bool = False) -> dict | None:
    """Create a folder. Returns folder dict or None if already exists."""
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                "INSERT INTO folders (user_id, name, is_system) VALUES ($1, $2, $3) RETURNING id, name, is_system",
                user_id, name, is_system,
            )
            return {"id": row["id"], "name": row["name"], "is_system": row["is_system"]}
        except asyncpg.UniqueViolationError:
            return None


async def get_user_folders(pool: asyncpg.Pool, user_id: int) -> list[dict]:
    """Return all folders for a user."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, is_system FROM folders WHERE user_id = $1 ORDER BY name",
            user_id,
        )
        return [{"id": r["id"], "name": r["name"], "is_system": r["is_system"]} for r in rows]


async def get_folder_by_name(pool: asyncpg.Pool, user_id: int, name: str) -> dict | None:
    """Get a folder by name for a user."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, is_system FROM folders WHERE user_id = $1 AND name = $2",
            user_id, name,
        )
        if row:
            return {"id": row["id"], "name": row["name"], "is_system": row["is_system"]}
        return None


async def get_folder_by_id(pool: asyncpg.Pool, folder_id: int) -> dict | None:
    """Get a folder by its ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, user_id, name, is_system FROM folders WHERE id = $1",
            folder_id,
        )
        if row:
            return {"id": row["id"], "user_id": row["user_id"], "name": row["name"], "is_system": row["is_system"]}
        return None


# --- Link DB helpers ---

async def add_link(pool: asyncpg.Pool, user_id: int, url: str, description: str, folder_id: int | None = None) -> int:
    """Insert a link and return its database ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO links (user_id, url, description, folder_id) VALUES ($1, $2, $3, $4) RETURNING id",
            user_id, url, description, folder_id,
        )
        return row["id"]


async def get_user_links(pool: asyncpg.Pool, user_id: int, folder_id: int | None = None) -> list[dict]:
    """Return all links for a given user, optionally filtered by folder."""
    async with pool.acquire() as conn:
        if folder_id is not None:
            rows = await conn.fetch(
                "SELECT id, url, description, folder_id FROM links WHERE user_id = $1 AND folder_id = $2 ORDER BY created_at DESC",
                user_id, folder_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT id, url, description, folder_id FROM links WHERE user_id = $1 ORDER BY created_at DESC",
                user_id,
            )
        return [{"id": r["id"], "url": r["url"], "description": r["description"], "folder_id": r["folder_id"]} for r in rows]


async def find_link_by_url(pool: asyncpg.Pool, user_id: int, url: str) -> dict | None:
    """Check if a URL already exists for a given user. Returns the link dict or None."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, url, description, folder_id FROM links WHERE user_id = $1 AND url = $2",
            user_id, url,
        )
        if row:
            return {"id": row["id"], "url": row["url"], "description": row["description"], "folder_id": row["folder_id"]}
        return None


async def get_links_by_ids(pool: asyncpg.Pool, link_ids: list[int]) -> list[dict]:
    """Fetch specific links by their IDs."""
    if not link_ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, url, description, folder_id FROM links WHERE id = ANY($1::int[]) ORDER BY id",
            link_ids,
        )
        return [{"id": r["id"], "url": r["url"], "description": r["description"], "folder_id": r["folder_id"]} for r in rows]


async def delete_link(pool: asyncpg.Pool, user_id: int, link_id: int) -> bool:
    """Delete a link. Returns True if deleted, False if not found or protected."""
    async with pool.acquire() as conn:
        # Check if link exists and belongs to user
        row = await conn.fetchrow(
            "SELECT l.id, f.is_system FROM links l LEFT JOIN folders f ON l.folder_id = f.id WHERE l.id = $1 AND l.user_id = $2",
            link_id, user_id,
        )
        if row is None:
            return False
        if row["is_system"]:
            return False  # Protected system link
        await conn.execute("DELETE FROM links WHERE id = $1", link_id)
        return True


async def get_link_with_folder(pool: asyncpg.Pool, link_id: int) -> dict | None:
    """Get a link with its folder info."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT l.id, l.url, l.description, l.folder_id, f.name as folder_name, f.is_system FROM links l LEFT JOIN folders f ON l.folder_id = f.id WHERE l.id = $1",
            link_id,
        )
        if row:
            return {
                "id": row["id"], "url": row["url"], "description": row["description"],
                "folder_id": row["folder_id"], "folder_name": row["folder_name"], "is_system": row["is_system"],
            }
        return None


# --- Seed system folder ---

SYSTEM_FOLDER_NAME = "System"
DEFAULT_FOLDER_NAME = "General"

SYSTEM_DEMO_LINKS = [
    ("https://docs.python.org/3/", "Official Python documentation"),
    ("https://www.postgresql.org/docs/", "PostgreSQL documentation"),
    ("https://docs.aiogram.dev/en/latest/", "aiogram Telegram bot framework docs"),
    ("https://docs.docker.com/", "Docker documentation"),
    ("https://fastapi.tiangolo.com/", "FastAPI web framework docs"),
]


async def seed_system_folder(pool: asyncpg.Pool) -> None:
    """Create the System folder with demo links for all users (user_id=0 as global system user)."""
    SYSTEM_USER_ID = 0  # System user ID for global system folder

    folder = await get_folder_by_name(pool, SYSTEM_USER_ID, SYSTEM_FOLDER_NAME)
    if folder is None:
        folder = await create_folder(pool, SYSTEM_USER_ID, SYSTEM_FOLDER_NAME, is_system=True)
        if folder:
            logger.info("Created system folder (id=%d)", folder["id"])

    if folder is None:
        return  # Already exists

    # Check if demo links already exist
    async with pool.acquire() as conn:
        existing_count = await conn.fetchval(
            "SELECT COUNT(*) FROM links WHERE user_id = $1 AND folder_id = $2",
            SYSTEM_USER_ID, folder["id"],
        )

    if existing_count >= len(SYSTEM_DEMO_LINKS):
        return  # Already seeded

    for url, desc in SYSTEM_DEMO_LINKS:
        await add_link(pool, SYSTEM_USER_ID, url, desc, folder["id"])

    logger.info("Seeded %d demo links in system folder", len(SYSTEM_DEMO_LINKS))


async def ensure_default_folder(pool: asyncpg.Pool, user_id: int) -> int:
    """Ensure the user has a 'General' folder and return its ID."""
    folder = await get_folder_by_name(pool, user_id, DEFAULT_FOLDER_NAME)
    if folder is None:
        folder = await create_folder(pool, user_id, DEFAULT_FOLDER_NAME)
        if folder:
            logger.info("Created default folder '%s' for user %d", DEFAULT_FOLDER_NAME, user_id)
    return folder["id"] if folder else None


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Async client for OpenAI-compatible LLM API with X-API-Key auth."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def ask(self, system_prompt: str, user_message: str) -> str:
        """Send a chat completion request and return the assistant's text reply."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract text from OpenAI-compatible response
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e


# ---------------------------------------------------------------------------
# TTL Cache for /find results
# ---------------------------------------------------------------------------

class FindCache:
    """Simple in-memory TTL cache for /find results."""

    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self._cache: dict[str, tuple[list[int], float]] = {}

    def _key(self, user_id: int, query: str, folder_id: int | None) -> str:
        return f"{user_id}:{query}:{folder_id}"

    def get(self, user_id: int, query: str, folder_id: int | None) -> list[int] | None:
        key = self._key(user_id, query, folder_id)
        if key in self._cache:
            ids, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                logger.info("Cache hit for key=%s", key)
                return ids
            else:
                del self._cache[key]
        return None

    def set(self, user_id: int, query: str, folder_id: int | None, ids: list[int]) -> None:
        key = self._key(user_id, query, folder_id)
        self._cache[key] = (ids, time.time())
        logger.info("Cache set for key=%s, ids=%s", key, ids)

    def invalidate_user(self, user_id: int) -> None:
        """Remove all cache entries for a user (e.g. after adding/deleting links)."""
        keys_to_delete = [k for k in self._cache if k.startswith(f"{user_id}:")]
        for k in keys_to_delete:
            del self._cache[k]
        if keys_to_delete:
            logger.info("Invalidated %d cache entries for user %d", len(keys_to_delete), user_id)


find_cache = FindCache()


# ---------------------------------------------------------------------------
# Parse /add arguments
# ---------------------------------------------------------------------------

def parse_add_args(text: str) -> tuple[str, str, str | None] | None:
    """
    Parse '/add https://example.com "my description" --folder Work' → (url, description, folder_name).
    Supports both quoted and unquoted descriptions.
    """
    # Extract --folder flag
    folder_name = None
    folder_match = re.search(r'--folder\s+(\S+)', text)
    if folder_match:
        folder_name = folder_match.group(1)
        text = text[:folder_match.start()] + text[folder_match.end():]

    # Pattern: URL followed by optional quoted description
    pattern = r'^(\S+)\s+"([^"]+)"$'
    match = re.match(pattern, text.strip())
    if match:
        return match.group(1), match.group(2), folder_name

    # Fallback: URL + unquoted description
    parts = text.strip().split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1], folder_name

    # Just a URL, no description
    if parts and re.match(r'https?://', parts[0], re.IGNORECASE):
        return parts[0], parts[0], folder_name

    return None


# ---------------------------------------------------------------------------
# Parse /find arguments
# ---------------------------------------------------------------------------

def parse_find_args(text: str) -> tuple[str, str | None]:
    """
    Parse '/find python tutorial --folder Work' → ('python tutorial', 'Work').
    """
    folder_name = None
    folder_match = re.search(r'--folder\s+(\S+)', text)
    if folder_match:
        folder_name = folder_match.group(1)
        text = text[:folder_match.start()] + text[folder_match.end():]

    return text.strip(), folder_name


# ---------------------------------------------------------------------------
# Parse LLM response — expect list of IDs
# ---------------------------------------------------------------------------

def parse_llm_ids(text: str) -> list[int]:
    """
    Extract integer IDs from LLM text response.
    Handles formats like:
        - "1, 5, 12"
        - "[1, 5, 12]"
        - "IDs: 1, 5, 12"
        - "1\n5\n12"
    """
    # Try JSON array first
    json_match = re.search(r'\[([^\]]+)\]', text)
    if json_match:
        try:
            ids = json.loads(json_match.group(0))
            return [int(x) for x in ids if str(x).isdigit()]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract all integers from text
    return [int(x) for x in re.findall(r'\b(\d+)\b', text) if int(x) > 0]


# ---------------------------------------------------------------------------
# Bot handlers
# ---------------------------------------------------------------------------

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message):
    """Handle /start command."""
    await message.answer(
        "👋 Hi! I'm a bot for saving and searching links.\n\n"
        "📂 <b>Folders:</b>\n"
        "  /folders — list your folders\n"
        "  /create <code>name</code> — create a folder\n"
        "  /folder <code>name</code> — show links in a folder\n\n"
        "📎 <b>/add</b> <code>url \"description\" [--folder name]</code> — save a link\n"
        "🔍 <b>/find</b> <code>query [--folder name]</code> — find relevant links\n"
        "🗑 <b>/delete</b> <code>id</code> — delete a link\n"
        "📋 <b>/list</b> — show all your links\n\n"
        "Example: <code>/add https://docs.python.org \"official Python docs\" --folder Work</code>",
        parse_mode="HTML",
    )


@router.message(Command("add"))
async def cmd_add(message: Message):
    """Handle /add command — save a link to the database."""
    global DB_POOL

    # Extract arguments after the command
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a URL and description.\n"
            "Example: <code>/add https://example.com \"my site\" --folder Work</code>",
            parse_mode="HTML",
        )
        return

    parsed = parse_add_args(args[1])
    if parsed is None:
        await message.answer(
            "❌ Could not parse URL and description.\n"
            "Example: <code>/add https://example.com \"my site\" --folder Work</code>",
            parse_mode="HTML",
        )
        return

    url, description, folder_name = parsed
    user_id = message.from_user.id

    # Resolve folder
    if folder_name:
        folder = await get_folder_by_name(DB_POOL, user_id, folder_name)
        if folder is None:
            await message.answer(
                f"❌ Folder '<code>{folder_name}</code>' not found.\n"
                f"Create it first with: <code>/create {folder_name}</code>",
                parse_mode="HTML",
            )
            return
        folder_id = folder["id"]
    else:
        folder_id = await ensure_default_folder(DB_POOL, user_id)

    # Check for duplicate URL
    existing = await find_link_by_url(DB_POOL, user_id, url)
    if existing:
        folder_info = ""
        if existing["folder_id"]:
            f = await get_folder_by_id(DB_POOL, existing["folder_id"])
            if f:
                folder_info = f" (folder: <code>{f['name']}</code>)"
        await message.answer(
            f"⚠️ This link is already in your collection{folder_info}: {existing['description']}",
            parse_mode="HTML",
        )
        return

    try:
        link_id = await add_link(DB_POOL, user_id, url, description, folder_id)
        folder = await get_folder_by_id(DB_POOL, folder_id)
        folder_text = f" | 📁 {folder['name']}" if folder else ""
        await message.answer(
            f"✅ Link saved!\n"
            f"🆔 ID: <code>{link_id}</code>{folder_text}\n"
            f"🔗 URL: <code>{url}</code>\n"
            f"📝 Description: {description}",
            parse_mode="HTML",
        )
        # Invalidate cache for this user
        find_cache.invalidate_user(user_id)
    except Exception as e:
        logger.exception("Error saving link")
        await message.answer(f"❌ Error saving link: {e}", parse_mode="HTML")


@router.message(Command("find"))
async def cmd_find(message: Message):
    """
    Handle /find command — search links using LLM.

    Flow:
        1. Parse query and optional --folder flag
        2. Check cache
        3. Fetch user's links from DB (optionally filtered by folder)
        4. Send descriptions + user query to LLM
        5. Parse LLM response to get relevant link IDs
        6. Fetch those links from DB and send to user
    """
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer("❌ Please provide a search query.\nExample: <code>/find how to learn python --folder Work</code>", parse_mode="HTML")
        return

    query, folder_name = parse_find_args(args[1])
    if not query:
        await message.answer("❌ Please provide a search query.", parse_mode="HTML")
        return

    user_id = message.from_user.id

    # Resolve folder
    folder_id = None
    folder_text = ""
    if folder_name:
        folder = await get_folder_by_name(DB_POOL, user_id, folder_name)
        if folder is None:
            await message.answer(
                f"❌ Folder '<code>{folder_name}</code>' not found.",
                parse_mode="HTML",
            )
            return
        folder_id = folder["id"]
        folder_text = f" in folder '<code>{folder_name}</code>'"

    # Check cache
    cached_ids = find_cache.get(user_id, query, folder_id)
    if cached_ids is not None:
        found_links = await get_links_by_ids(DB_POOL, cached_ids)
        if found_links:
            result_text = f"📚 Found {len(found_links)} relevant links for query '{query}'{folder_text} (cached):\n\n"
            for link in found_links:
                result_text += f"🔗 <a href=\"{link['url']}\">{link['url']}</a>\n"
                result_text += f"   📝 {link['description']}\n\n"
            await message.answer(result_text, disable_web_page_preview=True, parse_mode="HTML")
            return

    # Step 1: get all user's links (optionally filtered by folder)
    links = await get_user_links(DB_POOL, user_id, folder_id)
    if not links:
        folder_hint = f" in folder '<code>{folder_name}</code>'" if folder_name else ""
        await message.answer(
            f"📭 You don't have any saved links{folder_hint}.\n"
            f"Add a link with: <code>/add url \"description\"</code>",
            parse_mode="HTML",
        )
        return

    # Step 2: build descriptions list for LLM
    descriptions_list = [f"ID {link['id']}: {link['description']}" for link in links]
    descriptions_text = "\n".join(descriptions_list)

    top_k = min(3, len(links))

    system_prompt = (
        "You are a search assistant that finds relevant links. "
        "You will receive a list of link descriptions with their IDs and a user's search query. "
        f"Select up to {top_k} most relevant descriptions for this query. "
        "Return ONLY the IDs of the selected links as a JSON array, for example: [1, 5, 12]. "
        "If none of the links are relevant, return an empty array []. "
        "If there are duplicate descriptions or the same URLs with different IDs — select only one, the most relevant. "
        "Do not add any explanations, only the array of IDs."
    )
    user_message = (
        f"From the list of descriptions:\n{descriptions_text}\n\n"
        f"Select up to {top_k} most relevant for the query: '{query}'. "
        f"Return only their database IDs, or [] if nothing is relevant."
    )

    # Show progress
    progress_msg = await message.answer("🔍 Searching for relevant links...")

    try:
        # Step 3: ask LLM
        llm = LLMClient(LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL)
        llm_reply = await llm.ask(system_prompt, user_message)
        logger.info("LLM reply for query '%s': %s", query, llm_reply)

        # Step 4: parse IDs
        found_ids = parse_llm_ids(llm_reply)
        if not found_ids:
            await progress_msg.edit_text(
                f"🔗 Nothing found for query: '{query}'{folder_text}."
            )
            return

        # Cache the results
        find_cache.set(user_id, query, folder_id, found_ids)

        # Step 5: fetch actual links
        found_links = await get_links_by_ids(DB_POOL, found_ids)

        if not found_links:
            await progress_msg.edit_text(
                f"🔗 No relevant links found for query: '{query}'{folder_text}."
            )
            return

        # Step 6: format and send results
        result_text = f"📚 Found {len(found_links)} relevant links for query '{query}'{folder_text}:\n\n"
        for link in found_links:
            result_text += f"🔗 <a href=\"{link['url']}\">{link['url']}</a>\n"
            result_text += f"   📝 {link['description']}\n\n"

        await progress_msg.edit_text(result_text, disable_web_page_preview=True, parse_mode="HTML")

    except Exception as e:
        logger.exception("Error during /find")
        await progress_msg.edit_text(f"❌ Error during search: {e}")


@router.message(Command("list"))
async def cmd_list(message: Message):
    """Handle /list command — show all user's saved links."""
    global DB_POOL

    user_id = message.from_user.id
    links = await get_user_links(DB_POOL, user_id)

    if not links:
        await message.answer("📭 You don't have any saved links yet.", parse_mode="HTML")
        return

    # Group links by folder
    folder_map: dict[str, list[dict]] = {}
    for link in links:
        folder = await get_folder_by_id(DB_POOL, link["folder_id"]) if link["folder_id"] else None
        folder_name = folder["name"] if folder else "Unsorted"
        if folder_name not in folder_map:
            folder_map[folder_name] = []
        folder_map[folder_name].append(link)

    text = f"📋 Your saved links ({len(links)}):\n\n"
    for fname, flinks in sorted(folder_map.items()):
        text += f"📁 <b>{fname}</b> ({len(flinks)}):\n"
        for link in flinks:
            text += f"  🆔 {link['id']} | <a href=\"{link['url']}\">{link['description']}</a>\n"
        text += "\n"

    await message.answer(text, disable_web_page_preview=True, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Folder commands
# ---------------------------------------------------------------------------

@router.message(Command("folders"))
async def cmd_folders(message: Message):
    """Handle /folders command — list user's folders."""
    global DB_POOL

    user_id = message.from_user.id
    folders = await get_user_folders(DB_POOL, user_id)

    if not folders:
        await message.answer("📂 You don't have any folders yet.\nCreate one with: <code>/create name</code>", parse_mode="HTML")
        return

    text = "📂 Your folders:\n\n"
    for folder in folders:
        icon = "🔒" if folder["is_system"] else "📁"
        links_count = len(await get_user_links(DB_POOL, user_id, folder["id"]))
        text += f"{icon} <code>{folder['name']}</code> — {links_count} links\n"

    await message.answer(text, parse_mode="HTML")


@router.message(Command("create"))
async def cmd_create(message: Message):
    """Handle /create command — create a folder."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a folder name.\n"
            "Example: <code>/create Work</code>",
            parse_mode="HTML",
        )
        return

    name = args[1].strip()
    user_id = message.from_user.id

    # Validate folder name
    if not re.match(r'^[\w\s\-]+$', name):
        await message.answer(
            "❌ Folder name can only contain letters, numbers, spaces, and hyphens.",
            parse_mode="HTML",
        )
        return

    if len(name) > 50:
        await message.answer("❌ Folder name is too long (max 50 characters).", parse_mode="HTML")
        return

    folder = await create_folder(DB_POOL, user_id, name)
    if folder is None:
        await message.answer(f"📁 Folder '<code>{name}</code>' already exists.", parse_mode="HTML")
        return

    await message.answer(f"✅ Folder '<code>{name}</code>' created!", parse_mode="HTML")


@router.message(Command("folder"))
async def cmd_folder(message: Message):
    """Handle /folder command — show links in a specific folder."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a folder name.\n"
            "Example: <code>/folder Work</code>",
            parse_mode="HTML",
        )
        return

    name = args[1].strip()
    user_id = message.from_user.id

    folder = await get_folder_by_name(DB_POOL, user_id, name)
    if folder is None:
        await message.answer(f"❌ Folder '<code>{name}</code>' not found.", parse_mode="HTML")
        return

    links = await get_user_links(DB_POOL, user_id, folder["id"])
    if not links:
        await message.answer(
            f"📁 Folder '<code>{name}</code>' is empty.\n"
            f"Add a link: <code>/add url \"desc\" --folder {name}</code>",
            parse_mode="HTML",
        )
        return

    text = f"📁 <b>{name}</b> ({len(links)} links):\n\n"
    for link in links:
        text += f"🆔 {link['id']} | <a href=\"{link['url']}\">{link['description']}</a>\n"

    await message.answer(text, disable_web_page_preview=True, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Delete command
# ---------------------------------------------------------------------------

@router.message(Command("delete"))
async def cmd_delete(message: Message):
    """Handle /delete command — delete a link by ID."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a link ID.\n"
            "Example: <code>/delete 5</code>",
            parse_mode="HTML",
        )
        return

    try:
        link_id = int(args[1].strip())
    except ValueError:
        await message.answer("❌ Invalid link ID. Must be a number.", parse_mode="HTML")
        return

    user_id = message.from_user.id

    # Get link info before deleting
    link = await get_link_with_folder(DB_POOL, link_id)
    if link is None:
        await message.answer("❌ Link not found.", parse_mode="HTML")
        return

    # Check ownership
    if link.get("folder_id"):
        folder = await get_folder_by_id(DB_POOL, link["folder_id"])
        if folder and folder["user_id"] != user_id and folder["is_system"]:
            await message.answer(
                "🔒 Cannot delete links from the system folder.",
                parse_mode="HTML",
            )
            return

    deleted = await delete_link(DB_POOL, user_id, link_id)
    if not deleted:
        await message.answer(
            "🔒 Cannot delete this link (it may be from a protected folder).",
            parse_mode="HTML",
        )
        return

    folder_info = f" (folder: <code>{link['folder_name']}</code>)" if link.get("folder_name") else ""
    await message.answer(
        f"🗑 Link deleted{folder_info}:\n"
        f"🆔 ID: <code>{link_id}</code>\n"
        f"🔗 URL: <code>{link['url']}</code>",
        parse_mode="HTML",
    )
    # Invalidate cache for this user
    find_cache.invalidate_user(user_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    """Initialize DB pool, ensure schema, start polling."""
    global DB_POOL

    if not BOT_TOKEN:
        logger.error("BOT_TOKEN is not set. Set it in .env file.")
        sys.exit(1)

    # Initialize database
    DB_POOL = await init_db_pool()
    await ensure_schema(DB_POOL)

    # Seed system folder with demo links
    await seed_system_folder(DB_POOL)

    # Create bot and dispatcher
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Bot is starting. Polling for updates...")

    # Start polling — this blocks until the bot is stopped
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
