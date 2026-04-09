#!/usr/bin/env python3
"""
Telegram bot for saving and searching links via LLM-powered semantic search.

Commands:
    /add <url> "description" [--folder <name>]  — save a link with description
    /find <query> [--folder <name>]             — find relevant links using LLM
    /folders                                    — list your folders
    /create <name>                              — create a folder
    /folder <name>                              — show links in a folder
    /delete <url>                               — delete link(s) by URL
    /edit <url> <new description>               — edit link description
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
from mawo_pymorphy3 import create_analyzer

# Initialize morphological analyzer (takes 1-2 seconds on bot startup)
analyzer = create_analyzer()

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

SYSTEM_FOLDER_NAME = "System"
SYSTEM_USER_ID = 0  # Global system user


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
    """Return all folders for a user with link counts, sorted by created_at."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT f.id, f.name, f.is_system, f.created_at, COUNT(l.id) AS links_count
            FROM folders f
            LEFT JOIN links l ON f.id = l.folder_id
            WHERE f.user_id = $1
            GROUP BY f.id, f.name, f.is_system, f.created_at
            ORDER BY f.created_at ASC
        """, user_id)
        return [{
            "id": r["id"], "name": r["name"], "is_system": r["is_system"],
            "created_at": r["created_at"], "links_count": r["links_count"],
        } for r in rows]


async def get_system_folder(pool: asyncpg.Pool) -> dict | None:
    """Get the System folder with link count."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT f.id, f.name, f.is_system, f.created_at, COUNT(l.id) AS links_count
            FROM folders f
            LEFT JOIN links l ON f.id = l.folder_id
            WHERE f.user_id = $1 AND f.name = $2
            GROUP BY f.id, f.name, f.is_system, f.created_at
        """, SYSTEM_USER_ID, SYSTEM_FOLDER_NAME)
        if row:
            return {
                "id": row["id"], "name": row["name"], "is_system": row["is_system"],
                "created_at": row["created_at"], "links_count": row["links_count"],
            }
        return None


async def get_folder_links(pool: asyncpg.Pool, folder_id: int) -> list[dict]:
    """Get all links in a folder using JOIN."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT l.id, l.url, l.description, l.folder_id, l.created_at
            FROM links l
            INNER JOIN folders f ON l.folder_id = f.id
            WHERE f.id = $1
            ORDER BY l.created_at DESC
        """, folder_id)
        return [{
            "id": r["id"], "url": r["url"], "description": r["description"],
            "folder_id": r["folder_id"], "created_at": r["created_at"],
        } for r in rows]


async def get_all_user_links(pool: asyncpg.Pool, user_id: int) -> list[dict]:
    """Get ALL links for a user across all their folders using JOIN."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT l.id, l.url, l.description, l.folder_id, l.created_at, f.name AS folder_name
            FROM links l
            INNER JOIN folders f ON l.folder_id = f.id
            WHERE f.user_id = $1
            ORDER BY l.created_at DESC
        """, user_id)
        return [{
            "id": r["id"], "url": r["url"], "description": r["description"],
            "folder_id": r["folder_id"], "folder_name": r["folder_name"],
            "created_at": r["created_at"],
        } for r in rows]


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


async def delete_folder(pool: asyncpg.Pool, user_id: int, folder_id: int) -> tuple[bool, str]:
    """
    Delete a folder if it's empty (no links).
    Returns (success: bool, message: str).
    Cannot delete System folder.
    """
    async with pool.acquire() as conn:
        # Check if folder exists
        folder = await conn.fetchrow(
            "SELECT id, user_id, name, is_system FROM folders WHERE id = $1",
            folder_id,
        )
        
        if not folder:
            return False, "Folder not found"
        
        # Cannot delete System folder
        if folder["is_system"]:
            return False, "Cannot delete System folder"
        
        # Check if folder belongs to user
        if folder["user_id"] != user_id:
            return False, "Folder does not belong to you"
        
        # Check if folder has links
        link_count = await conn.fetchval(
            "SELECT COUNT(*) FROM links WHERE folder_id = $1",
            folder_id,
        )
        
        if link_count > 0:
            return False, f"Folder '{folder['name']}' has {link_count} link(s). Delete all links first"
        
        # Delete the folder
        await conn.execute(
            "DELETE FROM folders WHERE id = $1",
            folder_id,
        )
        
        return True, folder["name"]


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


async def delete_links_by_url(pool: asyncpg.Pool, user_id: int, url: str) -> int:
    """Delete all links matching URL for a user. Returns count of deleted links. Skips system links."""
    async with pool.acquire() as conn:
        # Find all matching links with system check
        rows = await conn.fetch("""
            SELECT l.id, f.is_system
            FROM links l
            LEFT JOIN folders f ON l.folder_id = f.id
            WHERE l.user_id = $1 AND l.url = $2
        """, user_id, url)

        if not rows:
            return 0

        deleted = 0
        for row in rows:
            if row["is_system"]:
                continue  # Skip system links
            await conn.execute("DELETE FROM links WHERE id = $1", row["id"])
            deleted += 1

        return deleted


async def get_links_by_url(pool: asyncpg.Pool, user_id: int, url: str) -> list[dict]:
    """Get all links matching a URL for a user, with folder info."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT l.id, l.url, l.description, l.folder_id, f.name as folder_name, f.is_system
            FROM links l
            LEFT JOIN folders f ON l.folder_id = f.id
            WHERE l.user_id = $1 AND l.url = $2
        """, user_id, url)
        return [{
            "id": r["id"], "url": r["url"], "description": r["description"],
            "folder_id": r["folder_id"], "folder_name": r["folder_name"], "is_system": r["is_system"],
        } for r in rows]


async def edit_link_description(pool: asyncpg.Pool, user_id: int, url: str, new_description: str) -> bool:
    """Edit description of a link by URL. Returns False if link is system-protected."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT l.id, f.is_system
            FROM links l
            LEFT JOIN folders f ON l.folder_id = f.id
            WHERE l.user_id = $1 AND l.url = $2
            LIMIT 1
        """, user_id, url)

        if row is None:
            return None  # Link not found
        if row["is_system"]:
            return False  # System link

        await conn.execute(
            "UPDATE links SET description = $1 WHERE user_id = $2 AND url = $3",
            new_description, user_id, url,
        )
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

SYSTEM_DEMO_LINKS = [
    ("https://docs.python.org/3/", "Official Python documentation"),
    ("https://www.postgresql.org/docs/", "PostgreSQL documentation"),
    ("https://docs.docker.com/", "Docker documentation"),
    ("https://fastapi.tiangolo.com/", "FastAPI web framework docs"),
    ("https://git-scm.com/doc", "Git documentation"),
    ("https://docs.aiogram.dev/en/latest/", "aiogram Telegram bot framework docs"),
    ("https://redis.io/docs/", "Redis documentation"),
    ("https://www.sqlite.org/docs.html", "SQLite documentation"),
    ("https://nodejs.org/en/docs/", "Node.js documentation"),
    ("https://react.dev/reference/react", "React documentation"),
]


async def seed_system_folder(pool: asyncpg.Pool) -> None:
    """Create the System folder with demo links (global, user_id=0)."""
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


async def ensure_user_folder(pool: asyncpg.Pool, user_id: int, username: str | None = None) -> int:
    """Ensure user has a personal folder named after their username or user_id. Returns folder ID."""
    folder_name = username or str(user_id)

    folder = await get_folder_by_name(pool, user_id, folder_name)
    if folder is None:
        folder = await create_folder(pool, user_id, folder_name)
        if folder:
            logger.info("Created user folder '%s' for user %d", folder_name, user_id)

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
        """Remove all cache entries for a user."""
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
    """
    folder_name = None
    folder_match = re.search(r'--folder\s+(\S+)', text)
    if folder_match:
        folder_name = folder_match.group(1)
        text = text[:folder_match.start()] + text[folder_match.end():]

    pattern = r'^(\S+)\s+"([^"]+)"$'
    match = re.match(pattern, text.strip())
    if match:
        return match.group(1), match.group(2), folder_name

    parts = text.strip().split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1], folder_name

    if parts and re.match(r'https?://', parts[0], re.IGNORECASE):
        return parts[0], parts[0], folder_name

    return None


# ---------------------------------------------------------------------------
# Parse /find arguments
# ---------------------------------------------------------------------------

def parse_find_args(text: str) -> tuple[str, str | None]:
    """Parse '/find python tutorial --folder Work' → ('python tutorial', 'Work')."""
    folder_name = None
    folder_match = re.search(r'--folder\s+(\S+)', text)
    if folder_match:
        folder_name = folder_match.group(1)
        text = text[:folder_match.start()] + text[folder_match.end():]

    return text.strip(), folder_name


# ---------------------------------------------------------------------------
# Parse /edit arguments
# ---------------------------------------------------------------------------

def parse_edit_args(text: str) -> tuple[str, str] | None:
    """
    Parse '/edit https://example.com "new description"' → (url, new_description).
    Also handles: /edit https://example.com new description (rest is description).
    """
    # Quoted description
    match = re.match(r'^(\S+)\s+"([^"]+)"$', text.strip())
    if match:
        return match.group(1), match.group(2)

    # Unquoted: first token is URL, rest is description
    parts = text.strip().split(maxsplit=1)
    if len(parts) == 2 and re.match(r'https?://', parts[0], re.IGNORECASE):
        return parts[0], parts[1]

    return None


# ---------------------------------------------------------------------------
# Parse LLM response — expect list of IDs
# ---------------------------------------------------------------------------

def parse_llm_ids(text: str) -> list[int]:
    """Extract integer IDs from LLM text response."""
    json_match = re.search(r'\[([^\]]+)\]', text)
    if json_match:
        try:
            ids = json.loads(json_match.group(0))
            return [int(x) for x in ids if str(x).isdigit()]
        except (json.JSONDecodeError, ValueError):
            pass

    return [int(x) for x in re.findall(r'\b(\d+)\b', text) if int(x) > 0]


# ---------------------------------------------------------------------------
# Morphological analysis helpers (Russian + English language support)
# ---------------------------------------------------------------------------

def normalize_english(word: str) -> str:
    """Простая нормализация английских слов (стемминг)."""
    word = word.lower()
    if len(word) < 3:
        return word
    
    # Убираем распространённые окончания
    if word.endswith('ing'):
        word = word[:-3]
    elif word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('ly'):
        word = word[:-2]
    elif word.endswith('tion'):
        word = word[:-4]
    elif word.endswith('s') and not word.endswith('ss'):
        word = word[:-1]
    
    return word


def normalize_word(word: str) -> str:
    """Приводит слово к нормальной форме (русский через pymorphy3, английский через стемминг)."""
    if not word or len(word) < 2:
        return word.lower()
    
    # Сначала пробуем русский через pymorphy3
    try:
        parsed = analyzer.parse(word)[0]
        return parsed.normal_form
    except Exception:
        pass
    
    # Если не получилось — пробуем английскую нормализацию
    return normalize_english(word)


def is_search_command(text: str) -> tuple[bool, str]:
    """Проверяет, является ли текст поисковым запросом.
    Возвращает (is_search, cleaned_query)."""
    # Удаляем знаки препинания перед обработкой
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()

    # Слова-триггеры поиска (в нормальной форме + стеммированные варианты)
    search_triggers = {
        'найти', 'искать', 'поискать', 'отыскать', 'разыскать',
        'search', 'searching', 'search',  # search, searching, searches
        'find', 'finding', 'find'  # find, finding, finds
    }

    # Удаляем стоп-слова (в нормальной форме)
    stop_words = {
        'пожалуйста', 'помоги', 'можешь', 'можно', 'хочу', 'нужно',
        'please', 'help', 'мне', 'тебе', 'ему', 'ей', 'быстро', 'давай',
        'где', 'поищи', 'покажи'
    }

    cleaned = []
    is_search = False

    for word in words:
        normal = normalize_word(word)
        # Проверяем и нормальную форму, и стеммированную
        if normal in search_triggers or normalize_english(word) in search_triggers:
            is_search = True
            continue  # не добавляем триггер в запрос
        if normal in stop_words:
            continue  # пропускаем стоп-слова
        cleaned.append(word)

    return is_search, ' '.join(cleaned)


def is_add_command(text: str) -> bool:
    """Проверяет, является ли текст командой добавления."""
    # Удаляем знаки препинания перед обработкой
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()
    
    add_triggers = {
        'добавить', 'сохранить', 'запомнить', 'записать',
        'add', 'adding', 'add',  # add, adding, adds
        'save', 'saving', 'save'  # save, saving, saves
    }

    for word in words:
        normal = normalize_word(word)
        # Проверяем и нормальную форму, и стеммированную
        if normal in add_triggers or normalize_english(word) in add_triggers:
            return True
    return False


def is_folder_command(text: str) -> tuple[bool, str | None]:
    """Проверяет команды работы с папками.
    Возвращает (is_command, folder_name или None)."""
    # Удаляем знаки препинания перед обработкой
    text_lower = re.sub(r'[^\w\s]', '', text.lower())

    # Список папок (поддержка разных форм)
    if any(phrase in text_lower for phrase in [
        'покажи папки', 'список папок', 'мои папки',
        'folders', 'folder list', 'list folders', 'show folders'
    ]):
        return True, "__LIST__"

    # Удаление папки (поддержка deleting, removed и т.д.)
    delete_match = re.search(
        r'(?:удали|удалить|delete|deleting|deleted|remove|removing|removed)\s+(?:папку|folder\s+)?(\w+)',
        text_lower
    )
    if delete_match:
        return True, f"__DELETE__{delete_match.group(1)}"

    # Создание папки (поддержка creating, created и т.д.)
    create_match = re.search(
        r'(?:создай|создать|create|creating|created|new|making|made)\s+(?:папку|folder\s+)?(\w+)',
        text_lower
    )
    if create_match:
        return True, f"__CREATE__{create_match.group(1)}"

    # Открытие папки (поддержка showing, opened и т.д.)
    open_match = re.search(
        r'(?:открой|открыть|покажи|show|showing|showed|open|opening|opened|view|viewing)\s+(?:папку|folder\s+)?(\w+)',
        text_lower
    )
    if open_match:
        return True, open_match.group(1)

    return False, None


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
        "  /folder <code>name</code> — show links in a folder\n"
        "  /delete_folder <code>name</code> — delete an empty folder\n\n"
        "📎 <b>/add</b> <code>url \"description\" [--folder name]</code> — save a link\n"
        "🔍 <b>/find</b> <code>query [--folder name]</code> — find relevant links\n"
        "🗑 <b>/delete</b> <code>url</code> — delete link(s) by URL\n"
        "✏️ <b>/edit</b> <code>url \"new description\"</code> — edit description\n"
        "📋 <b>/list</b> — show all your links\n\n"
        "💡 <b>Natural language:</b>\n"
        "  <code>покажи папки</code>, <code>удали папку Work</code>, <code>найди python</code>\n\n"
        "Example: <code>/add https://docs.python.org \"official Python docs\" --folder Work</code>",
        parse_mode="HTML",
    )


@router.message(Command("add"))
async def cmd_add(message: Message):
    """Handle /add command — save a link to the database."""
    global DB_POOL

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
            "Example: <code>/add https://example.com \"my site\"</code>",
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
        # Auto-create user folder on first link add
        username = message.from_user.username
        folder_id = await ensure_user_folder(DB_POOL, user_id, username)

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
        find_cache.invalidate_user(user_id)
    except Exception as e:
        logger.exception("Error saving link")
        await message.answer(f"❌ Error saving link: {e}", parse_mode="HTML")


async def cmd_find_logic(message: Message, query: str, folder_name: str | None = None):
    """
    Core find logic — search links using LLM.
    Searches across ALL user folders + System folder.
    """
    global DB_POOL

    user_id = message.from_user.id

    # Resolve folder filter (optional)
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

    # Gather links: user's links (all folders) + System links
    if folder_id is not None:
        # Filtered by specific folder
        links = await get_folder_links(DB_POOL, folder_id)
    else:
        # All user links across all folders + System links
        user_links = await get_all_user_links(DB_POOL, user_id)
        system_folder = await get_system_folder(DB_POOL)
        system_links = []
        if system_folder:
            system_links = await get_folder_links(DB_POOL, system_folder["id"])
        links = user_links + system_links

    if not links:
        folder_hint = f" in folder '<code>{folder_name}</code>'" if folder_name else ""
        await message.answer(
            f"📭 You don't have any saved links{folder_hint}.\n"
            f"Add a link with: <code>/add url \"description\"</code>",
            parse_mode="HTML",
        )
        return

    # Build descriptions list for LLM
    descriptions_list = [f"ID {link['id']}: {link['description']}" for link in links]
    descriptions_text = "\n".join(descriptions_list)

    top_k = min(3, len(links))

    system_prompt = (
        "You are a search assistant that finds relevant links. "
        "You will receive a list of link descriptions with their IDs and a user's search query. "
        f"Select up to {top_k} most relevant descriptions for this query. "
        "Return ONLY the IDs of the selected links as a JSON array, for example: [1, 5, 12]. "
        "If none of the links are relevant, return an empty array []. "
        "Do not add any explanations, only the array of IDs."
    )
    user_message = (
        f"From the list of descriptions:\n{descriptions_text}\n\n"
        f"Select up to {top_k} most relevant for the query: '{query}'. "
        f"Return only their database IDs, or [] if nothing is relevant."
    )

    progress_msg = await message.answer("🔍 Searching for relevant links...")

    try:
        llm = LLMClient(LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL)
        llm_reply = await llm.ask(system_prompt, user_message)
        logger.info("LLM reply for query '%s': %s", query, llm_reply)

        found_ids = parse_llm_ids(llm_reply)
        if not found_ids:
            await progress_msg.edit_text(
                f"🔗 Nothing found for query: '{query}'{folder_text}."
            )
            return

        find_cache.set(user_id, query, folder_id, found_ids)

        found_links = await get_links_by_ids(DB_POOL, found_ids)
        if not found_links:
            await progress_msg.edit_text(
                f"🔗 No relevant links found for query: '{query}'{folder_text}."
            )
            return

        result_text = f"📚 Found {len(found_links)} relevant links for query '{query}'{folder_text}:\n\n"
        for link in found_links:
            result_text += f"🔗 <a href=\"{link['url']}\">{link['url']}</a>\n"
            result_text += f"   📝 {link['description']}\n\n"

        await progress_msg.edit_text(result_text, disable_web_page_preview=True, parse_mode="HTML")

    except Exception as e:
        logger.exception("Error during /find")
        await progress_msg.edit_text(f"❌ Error during search: {e}")


@router.message(Command("find"))
async def cmd_find(message: Message):
    """
    Handle /find command — search links using LLM.
    Searches across ALL user folders + System folder.
    """
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer("❌ Please provide a search query.\nExample: <code>/find how to learn python</code>", parse_mode="HTML")
        return

    query, folder_name = parse_find_args(args[1])
    if not query:
        await message.answer("❌ Please provide a search query.", parse_mode="HTML")
        return

    await cmd_find_logic(message, query, folder_name)


@router.message(Command("list"))
async def cmd_list(message: Message):
    """Handle /list command — show all user's saved links grouped by folder."""
    global DB_POOL

    user_id = message.from_user.id

    # Get all user links (JOIN query) + system links
    user_links = await get_all_user_links(DB_POOL, user_id)
    system_folder = await get_system_folder(DB_POOL)
    system_links = []
    if system_folder:
        system_links = await get_folder_links(DB_POOL, system_folder["id"])

    all_links = user_links + system_links
    if not all_links:
        await message.answer("📭 You don't have any saved links yet.", parse_mode="HTML")
        return

    # Group by folder
    folder_map: dict[str, list[dict]] = {}
    for link in all_links:
        fname = link.get("folder_name") or "Unsorted"
        if fname not in folder_map:
            folder_map[fname] = []
        folder_map[fname].append(link)

    text = f"📋 Your saved links ({len(all_links)}):\n\n"
    for fname, flinks in sorted(folder_map.items()):
        icon = "🔒" if fname == SYSTEM_FOLDER_NAME else "📁"
        text += f"{icon} <b>{fname}</b> ({len(flinks)}):\n"
        for link in flinks:
            text += f"  🆔 {link['id']} | <a href=\"{link['url']}\">{link['description']}</a>\n"
        text += "\n"

    await message.answer(text, disable_web_page_preview=True, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Folder commands
# ---------------------------------------------------------------------------

@router.message(Command("folders"))
async def cmd_folders(message: Message):
    """Handle /folders command — list user's folders + System."""
    global DB_POOL

    user_id = message.from_user.id

    # Get user folders (single JOIN query with counts)
    folders = await get_user_folders(DB_POOL, user_id)

    # Also show System folder
    system_folder = await get_system_folder(DB_POOL)
    if system_folder:
        folders.append(system_folder)

    if not folders:
        await message.answer(
            "📂 You don't have any folders yet.\n"
            "Add a link to auto-create your folder: <code>/add url \"desc\"</code>\n"
            "Or create manually: <code>/create name</code>",
            parse_mode="HTML",
        )
        return

    text = "📂 Your folders:\n\n"
    for folder in folders:
        icon = "🔒" if folder["is_system"] else "📁"
        text += f"{icon} <code>{folder['name']}</code> — {folder['links_count']} links\n"

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
    """Handle /folder command — show links in a specific folder using JOIN."""
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

    # Look up folder: user's folders first, then System
    folder = await get_folder_by_name(DB_POOL, user_id, name)
    if folder is None and name == SYSTEM_FOLDER_NAME:
        folder = await get_system_folder(DB_POOL)

    if folder is None:
        await message.answer(f"❌ Folder '<code>{name}</code>' not found.", parse_mode="HTML")
        return

    # Get links with JOIN
    links = await get_folder_links(DB_POOL, folder["id"])
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
# Delete folder command
# ---------------------------------------------------------------------------

@router.message(Command("delete_folder"))
async def cmd_delete_folder(message: Message):
    """Handle /delete_folder command — delete an empty folder."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a folder name.\n"
            "Example: <code>/delete_folder Work</code>\n\n"
            "⚠️ Note: Can only delete empty folders (no links).",
            parse_mode="HTML",
        )
        return

    name = args[1].strip()
    user_id = message.from_user.id

    # Cannot delete System folder
    if name.lower() == SYSTEM_FOLDER_NAME.lower():
        await message.answer("🔒 Cannot delete the System folder.", parse_mode="HTML")
        return

    # Find the folder
    folder = await get_folder_by_name(DB_POOL, user_id, name)
    if folder is None:
        await message.answer(f"❌ Folder '<code>{name}</code>' not found.", parse_mode="HTML")
        return

    # Try to delete the folder
    success, msg = await delete_folder(DB_POOL, user_id, folder["id"])
    
    if success:
        await message.answer(f"✅ Folder '<code>{msg}</code>' deleted!", parse_mode="HTML")
    else:
        await message.answer(f"❌ {msg}", parse_mode="HTML")


# ---------------------------------------------------------------------------
# Delete command (by URL)
# ---------------------------------------------------------------------------

@router.message(Command("delete"))
async def cmd_delete(message: Message):
    """Handle /delete command — delete link(s) by URL."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a URL to delete.\n"
            "Example: <code>/delete https://example.com</code>",
            parse_mode="HTML",
        )
        return

    url = args[1].strip()
    user_id = message.from_user.id

    # Check if any links exist with this URL
    matching = await get_links_by_url(DB_POOL, user_id, url)
    if not matching:
        await message.answer("❌ No links found with this URL.", parse_mode="HTML")
        return

    # Check if all are system links
    all_system = all(m["is_system"] for m in matching)
    if all_system:
        await message.answer("🔒 Cannot delete links from the System folder.", parse_mode="HTML")
        return

    deleted = await delete_links_by_url(DB_POOL, user_id, url)
    if deleted == 0:
        await message.answer("🔒 Cannot delete these links (they may be from the System folder).", parse_mode="HTML")
        return

    await message.answer(
        f"🗑 Deleted {deleted} link(s) with URL:\n"
        f"<code>{url}</code>",
        parse_mode="HTML",
    )
    find_cache.invalidate_user(user_id)


# ---------------------------------------------------------------------------
# Edit command
# ---------------------------------------------------------------------------

@router.message(Command("edit"))
async def cmd_edit(message: Message):
    """Handle /edit command — change link description by URL."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a URL and new description.\n"
            "Example: <code>/edit https://example.com \"updated description\"</code>",
            parse_mode="HTML",
        )
        return

    parsed = parse_edit_args(args[1])
    if parsed is None:
        await message.answer(
            "❌ Could not parse URL and description.\n"
            "Example: <code>/edit https://example.com \"updated description\"</code>",
            parse_mode="HTML",
        )
        return

    url, new_description = parsed
    user_id = message.from_user.id

    result = await edit_link_description(DB_POOL, user_id, url, new_description)

    if result is None:
        await message.answer("❌ No link found with this URL.", parse_mode="HTML")
    elif result is False:
        await message.answer("🔒 Cannot modify System links.", parse_mode="HTML")
    else:
        await message.answer(
            f"✅ Description updated!\n"
            f"🔗 URL: <code>{url}</code>\n"
            f"📝 New: {new_description}",
            parse_mode="HTML",
        )
        find_cache.invalidate_user(user_id)


# ---------------------------------------------------------------------------
# Generic text message handler (messages without / commands)
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_text(message: Message):
    """Handle plain text messages (no / command) with morphological analysis."""
    if not message.text:
        return

    text = message.text.strip()
    logger.info(f"handle_text received: {text}")

    # =========================================================================
    # PRIORITY 1: Folder commands (list, create, delete, view)
    # =========================================================================
    is_folder, folder_action = is_folder_command(text)
    
    if is_folder:
        if folder_action == "__LIST__":
            # Show folder list
            logger.info("Folder list command detected via morphological analysis")
            await cmd_folders(message)
            return
        
        elif folder_action.startswith("__DELETE__"):
            # Delete folder
            folder_name = folder_action.replace("__DELETE__", "").capitalize()
            logger.info(f"Delete folder detected: {folder_name}")
            message.text = f"/delete_folder {folder_name}"
            await cmd_delete_folder(message)
            return
        
        elif folder_action.startswith("__CREATE__"):
            # Create folder
            folder_name = folder_action.replace("__CREATE__", "")
            logger.info(f"Create folder detected: {folder_name}")
            message.text = f"/create {folder_name}"
            await cmd_create(message)
            return
        
        else:
            # View folder
            folder_name = folder_action.capitalize()
            logger.info(f"View folder detected: {folder_name}")
            message.text = f"/folder {folder_name}"
            await cmd_folder(message)
            return

    # =========================================================================
    # PRIORITY 2: Add link commands
    # =========================================================================
    if is_add_command(text):
        logger.info("Add command detected via morphological analysis")
        # Extract URL if present
        url_match = re.search(r'https?://\S+', text)
        if url_match:
            url = url_match.group(0).rstrip('.,;:!?)')
            # Try to extract description (text after URL)
            url_pos = text.lower().find(url.lower())
            description = text[url_pos + len(url):].strip().strip('"\'')

            if description and len(description) > 3:
                message.text = f'/add {url} "{description}"'
                logger.info(f"Simulating /add with URL and description")
                await cmd_add(message)
                return

        await message.answer(
            "📎 To add a link, use:\n"
            "<code>/add https://example.com \"description\"</code>\n\n"
            "Or just send: <code>добавь https://example.com \"моё описание\"</code>",
            parse_mode="HTML",
        )
        return

    # =========================================================================
    # PRIORITY 3: Search commands
    # =========================================================================
    is_search, query = is_search_command(text)
    
    if is_search:
        logger.info(f"Search detected via morphological analysis, query: '{query}'")

        if query and len(query) > 1:
            await cmd_find_logic(message, query)
        else:
            await message.answer(
                "❌ Please provide a search query.\n"
                "Example: <code>find python tutorial</code> or <code>найди документация python</code>",
                parse_mode="HTML",
            )
        return

    # =========================================================================
    # FALLBACK: Show help
    # =========================================================================
    logger.info("No matching keywords found, showing help message")
    await message.answer(
        "👋 I understand natural language commands!\n\n"
        "🔍 <b>Search:</b>\n"
        "  <code>find python tutorial</code>\n"
        "  <code>найди документация python</code>\n\n"
        "📂 <b>Folders:</b>\n"
        "  <code>покажи папки</code> — list all folders\n"
        "  <code>открой Work</code> — show links in folder\n"
        "  <code>создай папку Learning</code> — create new folder\n"
        "  <code>удали папку Work</code> — delete empty folder\n\n"
        "📎 <b>Add links:</b>\n"
        "  <code>добавь https://example.com \"my site\"</code>\n\n"
        "Or use traditional commands: /add, /find, /folders, /list, /delete_folder"
    )


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

    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
