# Link Manager Bot

Telegram bot for saving, organizing and searching links with AI-powered semantic search.

## Demo
![Screenshot_20260409_230441_Telegram](https://github.com/user-attachments/assets/151b794f-cb65-433a-8130-25b1a37a5b5b)
![Screenshot_20260409_230451_Telegram](https://github.com/user-attachments/assets/a2bfe81c-bf9a-4b63-99a0-a01fecb0a807)



## Product Context

**End users:** Students and professionals who need to save, organize and quickly find links from chats, emails and websites.

**Problem:** Users lose useful links in messy chat histories. Existing bookmark managers lack intelligent search. Shared links get buried and forgotten.

**Solution:** A Telegram bot that saves links with descriptions, organizes them into folders, supports sharing with access keys, and finds links by meaning using AI — not just by keywords.

## Features

### Implemented
- Save links with custom descriptions
- Organize links into folders
- List all links and folders
- AI-powered semantic search using Qwen LLM
- Share folders with read or write access keys
- Join shared folders
- Revoke access (owner only)
- Edit and delete links
- Delete empty folders
- Natural language input (experimental)
- System folder with immutable links

### Not Yet Implemented
- Web interface
- Full natural language support for all commands
- Link preview generation
- Export/import links

## Usage

### Basic Commands
- `/add <url> "<description>"` — save a link
- `/find <query>` — search links with AI
- `/list` — show all saved links
- `/folders` — list all folders
- `/create <name>` — create a new folder
- `/folder <name>` — open a folder
- `/delete <url or id>` — delete a link

### Advanced Commands
- `/share <name>` — generate a share key for a folder
- `/join <key>` — join a shared folder
- `/revoke <name>` — revoke access (owner only)
- `/share_list <name>` — show who has access
- `/edit <url> "<new description>"` — edit link description
- `/delete_folder <name>` — delete an empty folder

### Natural Language (Experimental)
- `find python tutorial` — search links
- `add https://python.org docs` — save a link
- `show folders` — list folders
- `create folder Work` — create a folder

## Deployment

### Requirements
- Ubuntu 24.04 (or any Linux with Docker support)
- Docker and Docker Compose installed
- Git
- Qwen Code API proxy running on port 42005 (for LLM access)

### Step-by-step Deployment

1. **Clone the repository**
   ```bash
   git clone https://github.com/NoName12299/se-toolkit-hackathon.git
   cd se-toolkit-hackathon
