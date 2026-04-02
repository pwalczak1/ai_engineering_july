# Run the script on Windows (PowerShell) and open both repos in Cursor

This guide covers **(1)** running `auto_optimize_letters.py` on a Windows machine with PowerShell, and **(2)** opening a **multi-root workspace** in Cursor that includes this repo and `rag-llm-example`.

---

## 1. Run `auto_optimize_letters.py` on Windows

### Prerequisites

- **Python 3.10+** installed and available as `python` or `py` in PowerShell.
- An **OpenAI API key** (and, if your company requires it, the correct **API gateway base URL** configured in the script — ask IT).

### Clone or copy the repo

Put `ai_engineering_july` somewhere on disk, for example:

`C:\dev\ai_engineering_july`

All commands below assume that folder is your current directory.

### Create and activate a virtual environment

Open **PowerShell**, then:

```powershell
cd C:\dev\ai_engineering_july

python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If you see an error about **scripts disabled**, allow scripts for your user (one-time):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then run `.\venv\Scripts\Activate.ps1` again.

Your prompt should show `(venv)` when the environment is active.

### Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Set the API key

**Current session only:**

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Persist for future PowerShell sessions** (optional):

```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")
```

Restart the terminal after setting the user-level variable.

### Run the script

The script loads data with a path **relative to the project root** (`src/data/letter_counting.json`). Run it **from the `ai_engineering_july` folder** (not from inside `src`):

```powershell
cd C:\dev\ai_engineering_july
.\venv\Scripts\Activate.ps1
python .\src\auto_optimize_letters.py
```

Training calls the OpenAI API many times; expect **cost and runtime** to be non-trivial.

### Troubleshooting (Windows)

| Issue | What to try |
|--------|-------------|
| `python` not found | Use `py -3 -m venv venv` or install Python from [python.org](https://www.python.org/downloads/) and check “Add Python to PATH”. |
| `Activate.ps1` blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (see above). |
| `OPENAI_API_KEY` error | Export the variable in the **same** PowerShell window before `python ...`. |
| Corporate network blocks `api.openai.com` | Use the **API gateway** URL your manager/IT provides and configure `OpenAIClient(base_url=...)` in the script (see AdalFlow docs for `base_url`). |

---

## 2. Open both repos in one Cursor workspace

You want a single Cursor window with **two folders**:

- `ai_engineering_july` (this repo)
- `rag-llm-example` (the reference RAG / AdalFlow example)

Both folders should live **next to each other** on disk, for example:

```text
C:\dev\
  ai_engineering_july\
  rag-llm-example\
```

### Option A — Use a `.code-workspace` file (recommended)

1. Create a file **next to both repos**, e.g. `C:\dev\ai-two-repos.code-workspace`, with this content (adjust drive/path if needed):

```json
{
  "folders": [
    { "path": "ai_engineering_july" },
    { "path": "rag-llm-example" }
  ],
  "settings": {}
}
```

2. In **Cursor**: **File → Open Workspace from File…** and choose `ai-two-repos.code-workspace`.

You should see **two roots** in the sidebar: `ai_engineering_july` and `rag-llm-example`.

### Option B — Build the workspace from the UI

1. **File → Open Folder…** and open the first repo (e.g. `ai_engineering_july`).
2. **File → Add Folder to Workspace…** and add the second repo (`rag-llm-example`).
3. **File → Save Workspace As…** and save e.g. `ai-two-repos.code-workspace` so you can reopen the same layout later.

### Clone the second repo if you do not have it yet

```powershell
cd C:\dev
git clone https://github.com/YOUR_USERNAME/rag-llm-example.git
```

(Use your real remote URL.)

---

## Quick reference

| Task | Command / action |
|------|-------------------|
| Activate venv | `.\venv\Scripts\Activate.ps1` |
| Run script | `python .\src\auto_optimize_letters.py` from `ai_engineering_july` |
| Open both repos in Cursor | Open a `.code-workspace` with two `folders`, or **Add Folder to Workspace** |
