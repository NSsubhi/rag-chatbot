# Railway Environment Variables & Services Guide

## Services Setup

### **You need to create: 1 Service** ✅

**Main Service (FastAPI Backend)**
- Deploy the FastAPI application
- This is your main API service
- Railway will automatically detect Python from `requirements.txt` and use the `Procfile`

**Note:** The Streamlit frontend (`app/frontend.py`) is typically run locally or deployed separately if needed. For Railway, you usually just deploy the API backend.

---

## Environment Variables

Add these in your Railway project dashboard under the **Variables** tab.

### 🔴 **Required: At least ONE LLM configuration**

Choose ONE of the following options:

---

### **Option 1: OpenAI (Recommended for Production)**

```env
# OpenAI API Key (REQUIRED for Option 1)
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# OpenAI Model (OPTIONAL - defaults to gpt-3.5-turbo)
MODEL_NAME=gpt-3.5-turbo
# OR
MODEL_NAME=gpt-4
# OR
MODEL_NAME=gpt-4-turbo-preview

# OpenAI Embedding Model (OPTIONAL - defaults to text-embedding-ada-002)
EMBEDDING_MODEL=text-embedding-ada-002
# OR (newer models)
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_MODEL=text-embedding-3-large
```

**What you need:**
- ✅ `OPENAI_API_KEY` - Your OpenAI API key from https://platform.openai.com/api-keys
- ⚠️ `MODEL_NAME` - Optional, defaults to `gpt-3.5-turbo`
- ⚠️ `EMBEDDING_MODEL` - Optional, defaults to `text-embedding-ada-002`

---

### **Option 2: HuggingFace Inference API (Free Tier Available)**

```env
# HuggingFace API Token (REQUIRED for Option 2)
HUGGINGFACE_API_TOKEN=hf_your-actual-hf-token-here

# HuggingFace Model (OPTIONAL - defaults to google/flan-t5-base)
HUGGINGFACE_MODEL=google/flan-t5-base
# OR other models
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
HUGGINGFACE_MODEL=facebook/blenderbot-400M-distill
```

**What you need:**
- ✅ `HUGGINGFACE_API_TOKEN` - Get from https://huggingface.co/settings/tokens
- ⚠️ `HUGGINGFACE_MODEL` - Optional, defaults to `google/flan-t5-base`

**Note:** This uses HuggingFace API (requires internet). Embeddings will use free local models.

---

### **Option 3: HuggingFace Local Models (Fully Free, No API Key)**

```env
# Leave OPENAI_API_KEY empty or don't set it
# Leave HUGGINGFACE_API_TOKEN empty or don't set it

# Local Model Name (OPTIONAL - defaults to microsoft/DialoGPT-small)
LOCAL_MODEL=microsoft/DialoGPT-small
# OR
LOCAL_MODEL=distilgpt2
LOCAL_MODEL=gpt2
```

**What you need:**
- ✅ **Nothing!** Just don't set `OPENAI_API_KEY` or `HUGGINGFACE_API_TOKEN`
- ⚠️ `LOCAL_MODEL` - Optional, defaults to `microsoft/DialoGPT-small`

**Note:** Models download on first use (slower startup, but free). Embeddings use free local models.

---

### **Option 4: Ollama (NOT Recommended for Railway)**

```env
# Ollama Model (only if running Ollama locally or on Railway)
OLLAMA_MODEL=llama2
# OR
OLLAMA_MODEL=llama2:13b
OLLAMA_MODEL=mistral
```

**⚠️ Warning:** Ollama typically runs locally. For Railway, you'd need to set up Ollama as a separate service, which is complex.

---

### 🟡 **Optional: Storage & Paths**

```env
# ChromaDB Storage Path (OPTIONAL - defaults to ./chroma_db)
CHROMA_DB_PATH=./data/chroma

# Upload Directory (OPTIONAL - defaults to ./data/uploads)
UPLOAD_DIR=./data/uploads
```

**⚠️ Important:** Railway uses **ephemeral storage**, so data in these directories will be lost on restart unless you use a Railway Volume.

**For Persistence:**
1. Add a **Railway Volume** service
2. Mount it to `/app/data`
3. Set `CHROMA_DB_PATH=/app/data/chroma`

---

### 🟢 **Automatically Set (Don't Set Manually)**

These are automatically handled by Railway:

```env
# ❌ DON'T SET THESE - Railway sets them automatically
PORT=8000  # Railway sets this automatically
```

---

### 🔵 **Optional: API URL**

```env
# API URL (OPTIONAL - mainly for frontend if you deploy Streamlit separately)
API_URL=https://your-app.railway.app
```

Railway will provide your app URL automatically. Only set this if:
- You're deploying the Streamlit frontend separately
- You need to override the default URL

---

## Quick Setup Examples

### **Minimum Setup (Free HuggingFace Local Models)**

```env
# That's it! No API keys needed.
# The app will use free local HuggingFace models automatically.
```

### **Recommended Setup (OpenAI)**

```env
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
```

### **Full Setup with Persistence**

```env
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
CHROMA_DB_PATH=/app/data/chroma
UPLOAD_DIR=/app/data/uploads
```

*(Requires Railway Volume mounted at `/app/data`)*

---

## Summary Table

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | If using OpenAI | - | OpenAI API key |
| `MODEL_NAME` | No | `gpt-3.5-turbo` | OpenAI model name |
| `EMBEDDING_MODEL` | No | `text-embedding-ada-002` | OpenAI embedding model |
| `HUGGINGFACE_API_TOKEN` | If using HF API | - | HuggingFace API token |
| `HUGGINGFACE_MODEL` | No | `google/flan-t5-base` | HuggingFace model name |
| `LOCAL_MODEL` | No | `microsoft/DialoGPT-small` | Local HuggingFace model |
| `OLLAMA_MODEL` | No | `llama2` | Ollama model name |
| `CHROMA_DB_PATH` | No | `./chroma_db` | ChromaDB storage path |
| `UPLOAD_DIR` | No | `./data/uploads` | Upload directory |
| `API_URL` | No | Auto-detected | API base URL |
| `PORT` | No | Auto-set | Server port (Railway sets this) |

---

## Adding Environment Variables in Railway

1. Go to your Railway project dashboard
2. Click on your service (the FastAPI backend)
3. Go to the **Variables** tab
4. Click **+ New Variable**
5. Add each variable name and value
6. Click **Add**

Railway will automatically redeploy when you add/modify environment variables.

---

## Testing Your Setup

After deployment, test your API:

```bash
# Health check
curl https://your-app.railway.app/health

# Should return:
{
  "status": "healthy",
  "rag_ready": true/false,
  "llm_configured": true/false,
  "embeddings_configured": true/false
}
```
