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
import secrets
import string
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

        # Create shared_folders table (for sharing folders via access keys)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_folders (
                id          SERIAL PRIMARY KEY,
                folder_id   INT       NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                access_key  TEXT      NOT NULL UNIQUE,
                created_by  BIGINT    NOT NULL,
                mode        TEXT      DEFAULT 'read',
                created_at  TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_shared_folders_access_key ON shared_folders(access_key);
            CREATE INDEX IF NOT EXISTS idx_shared_folders_folder_id ON shared_folders(folder_id);
            CREATE INDEX IF NOT EXISTS idx_shared_folders_created_by ON shared_folders(created_by);
        """)

        # Create shared_access table (for tracking which users have access to shared folders)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_access (
                id          SERIAL PRIMARY KEY,
                folder_id   INT       NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                user_id     BIGINT    NOT NULL,
                granted_by  BIGINT    NOT NULL,
                mode        TEXT      DEFAULT 'read',
                granted_at  TIMESTAMP DEFAULT NOW(),
                UNIQUE(folder_id, user_id)
            );
            CREATE INDEX IF NOT EXISTS idx_shared_access_folder_id ON shared_access(folder_id);
            CREATE INDEX IF NOT EXISTS idx_shared_access_user_id ON shared_access(user_id);
            CREATE INDEX IF NOT EXISTS idx_shared_access_granted_by ON shared_access(granted_by);
        """)

        # Add mode column if it doesn't exist (migration for existing tables)
        await conn.execute("""
            ALTER TABLE shared_folders ADD COLUMN IF NOT EXISTS mode TEXT DEFAULT 'read';
            ALTER TABLE shared_access ADD COLUMN IF NOT EXISTS mode TEXT DEFAULT 'read';
        """)

        logger.info("Schema ensured (folders, links, shared_folders, shared_access tables exist, folder_id column present)")


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


# --- Shared folders DB helpers ---

def generate_share_key(folder_name: str) -> str:
    """Generate a share key: folder_name_abc123 (6 random alphanumeric chars)."""
    chars = string.ascii_lowercase + string.digits
    random_part = ''.join(secrets.choice(chars) for _ in range(6))
    return f"{folder_name}_{random_part}"


async def get_existing_share_key(pool: asyncpg.Pool, folder_id: int) -> dict | None:
    """Get existing share key for a folder, or None if not shared. Returns dict with access_key and mode."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT access_key, mode FROM shared_folders WHERE folder_id = $1",
            folder_id,
        )
        if row:
            return {"access_key": row["access_key"], "mode": row["mode"]}
        return None


async def create_share_key(pool: asyncpg.Pool, folder_id: int, created_by: int, access_key: str, mode: str = 'read') -> None:
    """Insert a new share key into shared_folders."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO shared_folders (folder_id, access_key, created_by, mode) VALUES ($1, $2, $3, $4)",
            folder_id, access_key, created_by, mode,
        )


async def get_folder_by_share_key(pool: asyncpg.Pool, access_key: str) -> dict | None:
    """Get folder info by share key."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT f.id, f.name, f.user_id as owner_id, sf.access_key, sf.created_by, sf.mode
            FROM shared_folders sf
            INNER JOIN folders f ON sf.folder_id = f.id
            WHERE sf.access_key = $1
        """, access_key)
        if row:
            return {
                "id": row["id"], "name": row["name"], "owner_id": row["owner_id"],
                "access_key": row["access_key"], "created_by": row["created_by"],
                "mode": row["mode"],
            }
        return None


async def grant_folder_access(pool: asyncpg.Pool, folder_id: int, user_id: int, granted_by: int, mode: str = 'read') -> bool:
    """Grant access to a shared folder. Returns True if granted, False if already exists."""
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                "INSERT INTO shared_access (folder_id, user_id, granted_by, mode) VALUES ($1, $2, $3, $4)",
                folder_id, user_id, granted_by, mode,
            )
            return True
        except asyncpg.UniqueViolationError:
            return False


async def get_user_access_mode(pool: asyncpg.Pool, folder_id: int, user_id: int) -> str | None:
    """Get user's access mode for a folder. Returns 'read', 'write', or None."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT mode FROM shared_access WHERE folder_id = $1 AND user_id = $2",
            folder_id, user_id,
        )
        return row["mode"] if row else None


async def has_folder_access(pool: asyncpg.Pool, folder_id: int, user_id: int) -> bool:
    """Check if user has access to a shared folder."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM shared_access WHERE folder_id = $1 AND user_id = $2",
            folder_id, user_id,
        )
        return row is not None


async def get_shared_folder_users(pool: asyncpg.Pool, folder_id: int) -> list[dict]:
    """Get all users with access to a shared folder."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT sa.user_id, sa.mode, sa.granted_at, f.name as folder_name
            FROM shared_access sa
            INNER JOIN folders f ON sa.folder_id = f.id
            WHERE sa.folder_id = $1
            ORDER BY sa.granted_at DESC
        """, folder_id)
        return [{
            "user_id": r["user_id"], "mode": r["mode"],
            "granted_at": r["granted_at"], "folder_name": r["folder_name"],
        } for r in rows]


async def revoke_share_key(pool: asyncpg.Pool, folder_id: int, user_id: int) -> tuple[bool, str]:
    """
    Revoke access to a shared folder.
    Returns (success: bool, message: str).
    Only the folder owner can revoke access.
    """
    async with pool.acquire() as conn:
        # Check if folder exists and get owner info
        folder = await conn.fetchrow(
            "SELECT id, user_id, name, is_system FROM folders WHERE id = $1",
            folder_id,
        )
        
        if not folder:
            return False, "Folder not found"
        
        # Cannot revoke System folder
        if folder["is_system"]:
            return False, "Cannot revoke System folder"
        
        # Check if caller is the folder owner
        if folder["user_id"] != user_id:
            return False, "Only the folder owner can revoke access"
        
        # Check if folder is actually shared
        share_record = await conn.fetchrow(
            "SELECT id FROM shared_folders WHERE folder_id = $1",
            folder_id,
        )
        
        if not share_record:
            return False, f"Folder '{folder['name']}' is not shared"
        
        # Delete all access records
        await conn.execute(
            "DELETE FROM shared_access WHERE folder_id = $1",
            folder_id,
        )
        
        # Delete the share key record
        await conn.execute(
            "DELETE FROM shared_folders WHERE folder_id = $1",
            folder_id,
        )
        
        return True, folder["name"]


async def get_user_shared_folders(pool: asyncpg.Pool, user_id: int) -> list[dict]:
    """Get all folders shared with a user."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT f.id, f.name, f.user_id as owner_id, sa.granted_by, sa.granted_at
            FROM shared_access sa
            INNER JOIN folders f ON sa.folder_id = f.id
            WHERE sa.user_id = $1
            ORDER BY sa.granted_at DESC
        """, user_id)
        return [{
            "id": r["id"], "name": r["name"], "owner_id": r["owner_id"],
            "granted_by": r["granted_by"], "granted_at": r["granted_at"],
        } for r in rows]


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
            SELECT l.id, l.user_id, l.url, l.description, l.folder_id, f.name as folder_name, f.is_system
            FROM links l
            LEFT JOIN folders f ON l.folder_id = f.id
            WHERE l.user_id = $1 AND l.url = $2
        """, user_id, url)
        return [{
            "id": r["id"], "user_id": r["user_id"], "url": r["url"], "description": r["description"],
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
        "👋 <b>Welcome to the Link Saver Bot!</b>\n\n"
        "Save, organize, and find your favorite links using AI-powered semantic search.\n"
        "Share folders with your team and collaborate on curated link collections.\n\n"
        "📖 <b>Quick Start:</b>\n"
        "  1. <code>/create Work</code> — Create your first folder\n"
        "  2. <code>/add https://example.com \"My site\" --folder Work</code> — Save a link\n"
        "  3. <code>/find python</code> — Search with AI\n"
        "  4. <code>/share Work</code> — Share with others\n\n"
        "📚 <b>Commands:</b> Use /help for full command list\n\n"
        "💡 <b>Tip:</b> You can also use natural language like\n"
        "  <code>find python tutorial</code> or <code>show folders</code>",
        parse_mode="HTML",
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command — show all available commands."""
    await message.answer(
        "📚 **Core Commands** (always work)\n\n"
        "📎 `/add <url> \"<description>\" [--folder <name>]` — Save a link\n"
        "🔍 `/find <query> [--folder <name>]` — Search links with AI\n"
        "📂 `/folders` — List all your folders\n"
        "📁 `/create <name>` — Create a new folder\n"
        "📂 `/folder <name>` — Open a folder\n"
        "📋 `/list` — Show all your links\n\n"
        "🔧 **Advanced Features** (commands only)\n"
        "🔗 `/share <name> [--write]` — Generate share key for a folder\n"
        "🔑 `/join <access_key>` — Join a shared folder\n"
        "📡 `/share_list <name>` — See who has access to a folder\n"
        "🚫 `/revoke <name>` — Revoke access (owner only)\n"
        "🗑 `/delete_folder <name>` — Delete an empty folder\n"
        "✏️ `/edit <url> \"<new description>\"` — Edit link description\n"
        "🗑 `/delete <url>` — Delete a link by URL\n\n"
        "🤖 **Natural Language** (experimental)\n"
        "• `find python tutorial` — Search links\n"
        "• `add https://python.org docs` — Save a link\n"
        "• `show all my links` — List all links\n"
        "• `show folders` — List folders\n"
        "• `create folder Work` — Create a folder\n"
        "• `delete folder Old` — Delete a folder\n\n"
        "💡 **Tip:** For complex operations (sharing, editing, deleting),\n"
        "use the `/commands` above. For quick actions, just type naturally!",
        parse_mode="Markdown",
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
        is_shared_folder = False

        # If not found in own folders, check shared folders
        if folder is None:
            shared_folders_list = await get_user_shared_folders(DB_POOL, user_id)
            for sf in shared_folders_list:
                if sf["name"].lower() == folder_name.lower():
                    # Check if user has access
                    has_access = await has_folder_access(DB_POOL, sf["id"], user_id)
                    if has_access:
                        folder = sf
                        is_shared_folder = True
                    break

        if folder is None:
            await message.answer(
                f"❌ Folder '<code>{folder_name}</code>' not found.\n"
                f"Create it first with: <code>/create {folder_name}</code>",
                parse_mode="HTML",
            )
            return

        folder_id = folder["id"]
        folder_is_shared = is_shared_folder
        
        # Check if user has write access for shared folders
        if folder_is_shared:
            access_mode = await get_user_access_mode(DB_POOL, folder_id, user_id)
            if access_mode != 'write':
                await message.answer(
                    f"❌ You only have read access to folder '<code>{folder_name}</code>'.\n"
                    f"Ask the owner to grant you write access.",
                    parse_mode="HTML",
                )
                return
    else:
        # Auto-create user folder on first link add
        username = message.from_user.username
        folder_id = await ensure_user_folder(DB_POOL, user_id, username)
        folder_is_shared = False

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
        if folder:
            icon = "🔗" if folder_is_shared else "📁"
            folder_text = f" | {icon} {folder['name']}"
        else:
            folder_text = ""
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
    Searches across ALL user folders + shared folders + System folder.
    """
    global DB_POOL

    user_id = message.from_user.id

    # Resolve folder filter (optional)
    folder_id = None
    folder_text = ""
    if folder_name:
        folder = await get_folder_by_name(DB_POOL, user_id, folder_name)
        
        # If not found in own folders, check shared folders
        if folder is None:
            shared_folders_list = await get_user_shared_folders(DB_POOL, user_id)
            for sf in shared_folders_list:
                if sf["name"].lower() == folder_name.lower():
                    has_access = await has_folder_access(DB_POOL, sf["id"], user_id)
                    if has_access:
                        folder = sf
                    break
        
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

    # Gather links: user's links (all folders) + shared folders + System links
    if folder_id is not None:
        # Filtered by specific folder
        links = await get_folder_links(DB_POOL, folder_id)
    else:
        # All user links across all folders + shared folders + System links
        user_links = await get_all_user_links(DB_POOL, user_id)
        
        # Add links from shared folders
        shared_folders_list = await get_user_shared_folders(DB_POOL, user_id)
        shared_links = []
        for sf in shared_folders_list:
            sf_links = await get_folder_links(DB_POOL, sf["id"])
            shared_links.extend(sf_links)
        
        # System links
        system_folder = await get_system_folder(DB_POOL)
        system_links = []
        if system_folder:
            system_links = await get_folder_links(DB_POOL, system_folder["id"])
        
        links = user_links + shared_links + system_links

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
    """Handle /list command — show all user's saved links grouped by folder (own + shared + system)."""
    global DB_POOL

    user_id = message.from_user.id

    # Get all user links (JOIN query) + shared folder links + system links
    user_links = await get_all_user_links(DB_POOL, user_id)
    
    # Add links from shared folders
    shared_folders_list = await get_user_shared_folders(DB_POOL, user_id)
    shared_links = []
    for sf in shared_folders_list:
        sf_links = await get_folder_links(DB_POOL, sf["id"])
        # Mark shared links with folder info
        for link in sf_links:
            link["is_shared"] = True
            link["folder_name"] = sf["name"]
        shared_links.extend(sf_links)
    
    # System links
    system_folder = await get_system_folder(DB_POOL)
    system_links = []
    if system_folder:
        system_links = await get_folder_links(DB_POOL, system_folder["id"])
        for link in system_links:
            link["is_system"] = True
            link["folder_name"] = SYSTEM_FOLDER_NAME

    all_links = user_links + shared_links + system_links
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
        if fname == SYSTEM_FOLDER_NAME:
            icon = "🔒"
        elif flinks[0].get("is_shared"):
            icon = "🔗"
        else:
            icon = "📁"
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
    """Handle /folders command — list user's folders + shared + System."""
    global DB_POOL

    user_id = message.from_user.id

    # Get user's own folders (single JOIN query with counts)
    own_folders = await get_user_folders(DB_POOL, user_id)

    # Get shared folders (folders shared with this user)
    shared_folders_list = await get_user_shared_folders(DB_POOL, user_id)

    # Also show System folder
    system_folder = await get_system_folder(DB_POOL)

    # Check if we have any folders at all
    has_any = own_folders or shared_folders_list or system_folder
    if not has_any:
        await message.answer(
            "📂 You don't have any folders yet.\n"
            "Add a link to auto-create your folder: <code>/add url \"desc\"</code>\n"
            "Or create manually: <code>/create name</code>\n"
            "Or join a shared folder: <code>/join access_key</code>",
            parse_mode="HTML",
        )
        return

    text = "📂 Your folders:\n\n"

    # Own folders
    if own_folders:
        text += "📁 <b>My folders:</b>\n"
        for folder in own_folders:
            text += f"  📁 <code>{folder['name']}</code> — {folder['links_count']} links\n"
        text += "\n"

    # Shared folders
    if shared_folders_list:
        text += "🔗 <b>Shared with me:</b>\n"
        for folder in shared_folders_list:
            text += f"  🔗 <code>{folder['name']}</code> (shared)\n"
        text += "\n"

    # System folder
    if system_folder:
        text += "🔒 <b>System:</b>\n"
        text += f"  🔒 <code>{system_folder['name']}</code> — {system_folder['links_count']} links\n"

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
    is_shared = False
    
    if folder is None and name == SYSTEM_FOLDER_NAME:
        folder = await get_system_folder(DB_POOL)
    
    # If not found in own folders, check shared folders
    if folder is None:
        shared_folders_list = await get_user_shared_folders(DB_POOL, user_id)
        for sf in shared_folders_list:
            if sf["name"].lower() == name.lower():
                folder = sf
                is_shared = True
                break

    if folder is None:
        await message.answer(f"❌ Folder '<code>{name}</code>' not found.", parse_mode="HTML")
        return

    # Check access for shared folders
    if is_shared:
        has_access = await has_folder_access(DB_POOL, folder["id"], user_id)
        if not has_access:
            await message.answer(
                f"❌ You don't have access to folder '<code>{name}</code>'.\n"
                f"Ask the owner to share it with you.",
                parse_mode="HTML",
            )
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

    # Choose icon based on folder type
    if folder.get("is_system"):
        icon = "🔒"
    elif is_shared:
        icon = "🔗"
    else:
        icon = "📁"

    text = f"{icon} <b>{name}</b> ({len(links)} links):\n\n"
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
# Share folder command
# ---------------------------------------------------------------------------

@router.message(Command("share"))
async def cmd_share(message: Message):
    """Handle /share command — generate or show share key for a folder."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a folder name to share.\n"
            "Example: <code>/share Work</code> (read-only)\n"
            "Example: <code>/share Work --write</code> (read+write)",
            parse_mode="HTML",
        )
        return

    full_args = args[1].strip()
    
    # Parse --write flag
    mode = 'read'
    if '--write' in full_args.lower():
        mode = 'write'
        full_args = re.sub(r'--write', '', full_args, flags=re.IGNORECASE).strip()
    
    name = full_args.strip()
    user_id = message.from_user.id

    # Cannot share System folder
    if name.lower() == SYSTEM_FOLDER_NAME.lower():
        await message.answer("🔒 Cannot share the System folder.", parse_mode="HTML")
        return

    # Find the folder
    folder = await get_folder_by_name(DB_POOL, user_id, name)
    if folder is None:
        await message.answer(f"❌ Folder '<code>{name}</code>' not found.", parse_mode="HTML")
        return

    # Check if folder already has a share key
    existing_key_info = await get_existing_share_key(DB_POOL, folder["id"])
    if existing_key_info:
        mode_icon = "📖" if existing_key_info["mode"] == "read" else "✏️"
        await message.answer(
            f"🔗 Folder '<code>{folder['name']}</code>' is already shared!\n"
            f"{mode_icon} Share key: <code>{existing_key_info['access_key']}</code>\n"
            f"🔐 Mode: <code>{existing_key_info['mode']}</code>\n\n"
            f"Share this key with others so they can access the folder.",
            parse_mode="HTML",
        )
        return

    # Generate new share key
    access_key = generate_share_key(folder["name"])
    await create_share_key(DB_POOL, folder["id"], user_id, access_key, mode)

    mode_icon = "📖" if mode == "read" else "✏️"
    mode_text = "read-only" if mode == "read" else "read+write"
    
    await message.answer(
        f"✅ Folder '<code>{folder['name']}</code>' is now shared!\n"
        f"{mode_icon} Share key: <code>{access_key}</code>\n"
        f"🔐 Mode: <code>{mode_text}</code>\n\n"
        f"Share this key with others so they can access the folder.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Join shared folder command
# ---------------------------------------------------------------------------

@router.message(Command("join"))
async def cmd_join(message: Message):
    """Handle /join command — join a shared folder using an access key."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a share key to join.\n"
            "Example: <code>/join Work_x7k9m2</code>",
            parse_mode="HTML",
        )
        return

    access_key = args[1].strip()
    user_id = message.from_user.id

    # Find folder by share key
    folder_info = await get_folder_by_share_key(DB_POOL, access_key)
    if folder_info is None:
        await message.answer(
            "❌ Invalid or expired share key.\n"
            "Please check the key and try again.",
            parse_mode="HTML",
        )
        return

    # Check if user already has access
    has_access = await has_folder_access(DB_POOL, folder_info["id"], user_id)
    if has_access:
        await message.answer(
            f"ℹ️ You already have access to folder '<code>{folder_info['name']}</code>'.",
            parse_mode="HTML",
        )
        return

    # Grant access with the same mode as the share key
    await grant_folder_access(DB_POOL, folder_info["id"], user_id, folder_info["created_by"], folder_info["mode"])

    mode_icon = "📖" if folder_info["mode"] == "read" else "✏️"
    mode_text = "read-only" if folder_info["mode"] == "read" else "read+write"

    await message.answer(
        f"✅ Доступ к папке '<code>{folder_info['name']}</code>' получен!\n"
        f"{mode_icon} Mode: <code>{mode_text}</code>\n\n"
        f"You can now view it with: <code>/folder {folder_info['name']}</code>",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Revoke shared folder access command
# ---------------------------------------------------------------------------

@router.message(Command("revoke"))
async def cmd_revoke(message: Message):
    """Handle /revoke command — revoke access to a shared folder (owner only)."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a folder name to revoke.\n"
            "Example: <code>/revoke Work</code>",
            parse_mode="HTML",
        )
        return

    name = args[1].strip()
    user_id = message.from_user.id

    # Cannot revoke System folder
    if name.lower() == SYSTEM_FOLDER_NAME.lower():
        await message.answer("🔒 Cannot revoke access to the System folder.", parse_mode="HTML")
        return

    # Find the folder
    folder = await get_folder_by_name(DB_POOL, user_id, name)
    if folder is None:
        await message.answer(f"❌ Folder '<code>{name}</code>' not found.", parse_mode="HTML")
        return

    # Try to revoke the share key
    success, msg = await revoke_share_key(DB_POOL, folder["id"], user_id)

    if success:
        await message.answer(
            f"🔒 Access revoked for folder '<code>{msg}</code>'. Shared links are no longer available.",
            parse_mode="HTML",
        )
    else:
        await message.answer(f"❌ {msg}", parse_mode="HTML")


# ---------------------------------------------------------------------------
# Share list command
# ---------------------------------------------------------------------------

@router.message(Command("share_list"))
async def cmd_share_list(message: Message):
    """Handle /share_list command — show who has access to a shared folder."""
    global DB_POOL

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "❌ Please provide a folder name.\n"
            "Example: <code>/share_list Work</code>",
            parse_mode="HTML",
        )
        return

    name = args[1].strip()
    user_id = message.from_user.id

    # Cannot list System folder
    if name.lower() == SYSTEM_FOLDER_NAME.lower():
        await message.answer("🔒 Cannot list access for the System folder.", parse_mode="HTML")
        return

    # Find the folder
    folder = await get_folder_by_name(DB_POOL, user_id, name)
    if folder is None:
        await message.answer(f"❌ Folder '<code>{name}</code>' not found.", parse_mode="HTML")
        return

    # Check if folder is shared
    share_key_info = await get_existing_share_key(DB_POOL, folder["id"])
    if not share_key_info:
        await message.answer(
            f"❌ Folder '<code>{name}</code>' is not shared.\n"
            f"Share it first with: <code>/share {name}</code>",
            parse_mode="HTML",
        )
        return

    # Get all users with access
    users = await get_shared_folder_users(DB_POOL, folder["id"])

    if not users:
        await message.answer(
            f"🔗 Folder '<code>{name}</code>' is shared, but no one has joined yet.\n"
            f"🔑 Share key: <code>{share_key_info['access_key']}</code>",
            parse_mode="HTML",
        )
        return

    # Build response
    mode_icon_map = {"read": "📖", "write": "✏️"}
    text = f"🔗 Users with access to '<code>{name}</code>':\n\n"
    
    for user in users:
        icon = mode_icon_map.get(user["mode"], "❓")
        text += f"{icon} User ID: <code>{user['user_id']}</code> — {user['mode']}\n"
    
    text += f"\n🔑 Share key: <code>{share_key_info['access_key']}</code>"

    await message.answer(text, parse_mode="HTML")


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

    # Filter links based on ownership and access rights
    deletable_links = []
    for link in matching:
        # Skip system links
        if link["is_system"]:
            continue
        
        # Check if it's the user's own link (user_id matches)
        if link.get("user_id") == user_id:
            deletable_links.append(link)
            continue
        
        # For shared folders, check if user is the folder owner
        folder = await get_folder_by_id(DB_POOL, link["folder_id"])
        if folder and folder["user_id"] == user_id:
            # User owns the folder, can delete any link in it
            deletable_links.append(link)
            continue
        
        # For shared folders with write access, can only delete own links
        if folder:
            access_mode = await get_user_access_mode(DB_POOL, folder["id"], user_id)
            if access_mode == 'write' and link.get("user_id") == user_id:
                deletable_links.append(link)

    if not deletable_links:
        await message.answer(
            "🔒 Cannot delete these links (they belong to other users or you don't have permission).",
            parse_mode="HTML",
        )
        return

    # Delete only the links the user has permission to delete
    deleted = 0
    for link in deletable_links:
        await delete_links_by_url(DB_POOL, user_id, url)
        deleted += 1
        break  # delete_links_by_url already handles multiple links with same URL

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
# LLM-based intent routing
# ---------------------------------------------------------------------------

async def route_intent(text: str) -> dict:
    """Отправляет текст в LLM, получает JSON с intent и entities."""
    system_prompt = """Ты — роутер команд для Telegram бота. Сначала определи, есть ли в сообщении слово-триггер.

ШАГ 1: Поиск триггеров (синонимов команд):
- Поиск: ['найди', 'ищу', 'search', 'find', 'lookup', 'where']
- Добавление: ['добавь', 'сохрани', 'add', 'save', 'store', 'keep', 'bookmark']
- Список ссылок: ['покажи ссылки', 'list links', 'show all', 'my links']
- Папки: ['папки', 'folders', 'list folders']

ШАГ 2: Если триггер найден, извлеки сущности:
- Для добавления: найди URL (https?://\S+) и описание (всё после URL)
- Для поиска: всё, что после триггера

ШАГ 3: Верни JSON.

Примеры:
1. "add https://python.org docs" → {"intent": "add", "entities": {"url": "https://python.org", "description": "docs"}}
2. "save this link https://github.com" → {"intent": "add", "entities": {"url": "https://github.com", "description": "this link"}}
3. "find python tutorial" → {"intent": "search", "entities": {"query": "python tutorial"}}
4. "show all my links" → {"intent": "list_links", "entities": {}}
5. "python tutorial" → {"intent": "search", "entities": {"query": "python tutorial"}}
6. "покажи папки" → {"intent": "list_folders", "entities": {}}
7. "создай папку Work" → {"intent": "create_folder", "entities": {"name": "Work"}}

Верни ТОЛЬКО JSON, без пояснений."""

    llm = LLMClient(LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL)
    response = await llm.ask(system_prompt, f"Текст пользователя: {text}")
    return json.loads(response)


# ---------------------------------------------------------------------------
# Generic text message handler (messages without / commands)
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_text(message: Message):
    """Handle plain text messages (no / command) with LLM-based intent routing."""
    if not message.text:
        return
    
    text = message.text.strip()
    logger.info(f"handle_text received: {text}")
    
    try:
        intent_data = await route_intent(text)
        intent = intent_data.get("intent")
        entities = intent_data.get("entities", {})
        
        logger.info(f"LLM intent: {intent}, entities: {entities}")
        
        if intent == "search":
            query = entities.get("query", text)
            await cmd_find_logic(message, query)
            
        elif intent == "add":
            url = entities.get("url", "")
            description = entities.get("description", "")
            folder = entities.get("folder", "")
            
            if not url or not description:
                await message.answer(
                    "📎 Please provide URL and description.\n"
                    "Example: <code>/add https://example.com \"my site\"</code>",
                    parse_mode="HTML",
                )
                return
            
            # Construct command for existing handler
            if folder:
                message.text = f'/add {url} "{description}" --folder {folder}'
            else:
                message.text = f'/add {url} "{description}"'
            await cmd_add(message)
            
        elif intent == "list_folders":
            await cmd_folders(message)

        elif intent == "list_links":
            await cmd_list(message)

        elif intent == "create_folder":
            name = entities.get("name", "")
            if not name:
                await message.answer("❌ Please provide a folder name.", parse_mode="HTML")
                return
            message.text = f"/create {name}"
            await cmd_create(message)
            
        elif intent == "share":
            folder_name = entities.get("folder_name", "")
            if not folder_name:
                await message.answer("❌ Please provide a folder name to share.", parse_mode="HTML")
                return
            message.text = f"/share {folder_name}"
            await cmd_share(message)
            
        elif intent == "join":
            access_key = entities.get("access_key", "")
            if not access_key:
                await message.answer("❌ Please provide an access key.", parse_mode="HTML")
                return
            message.text = f"/join {access_key}"
            await cmd_join(message)
            
        elif intent == "delete_link":
            if "link_id" in entities:
                await message.answer(
                    "❌ Deletion by ID is not supported. Please use URL.\n"
                    "Example: <code>/delete https://example.com</code>",
                    parse_mode="HTML",
                )
            else:
                url = entities.get("url", "")
                if not url:
                    await message.answer("❌ Please provide a URL to delete.", parse_mode="HTML")
                    return
                message.text = f"/delete {url}"
                await cmd_delete(message)
            
        elif intent == "delete_folder":
            folder_name = entities.get("name", "")
            if not folder_name:
                await message.answer("❌ Please provide a folder name.", parse_mode="HTML")
                return
            message.text = f"/delete_folder {folder_name}"
            await cmd_delete_folder(message)
            
        elif intent == "edit":
            url = entities.get("url", "")
            new_description = entities.get("new_description", "")
            
            if not url or not new_description:
                await message.answer(
                    "❌ Please provide URL and new description.\n"
                    "Example: <code>/edit https://example.com \"updated description\"</code>",
                    parse_mode="HTML",
                )
                return
            message.text = f'/edit {url} "{new_description}"'
            await cmd_edit(message)
            
        elif intent == "help":
            await cmd_help(message)
            
        else:
            logger.warning(f"Unknown intent: {intent}")
            await cmd_help(message)
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        await message.answer(
            "❌ Sorry, I couldn't understand that. Please try again or use /help.",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error(f"Intent routing failed: {e}")
        await message.answer("Sorry, I didn't understand. Try /help")


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
