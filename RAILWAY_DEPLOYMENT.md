# Railway Deployment Guide

This guide will help you deploy the RAG Chatbot to Railway.

## Prerequisites

1. A Railway account (sign up at https://railway.app)
2. Git repository with your code (GitHub, GitLab, or Bitbucket)
3. Your API keys ready (OpenAI, HuggingFace, etc.)

## Deployment Steps

### 1. Push Your Code to Git

Make sure your code is in a Git repository:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Create a New Railway Project

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo" (or your Git provider)
4. Select your repository

### 3. Configure Environment Variables

In your Railway project dashboard:

1. Go to the "Variables" tab
2. Add the following environment variables:

**Required (at least one LLM configuration):**

```env
# Option 1: OpenAI
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# Option 2: HuggingFace (free)
# Leave OPENAI_API_KEY empty or don't set it
# The system will use HuggingFace models automatically

# Option 3: HuggingFace Inference API
HUGGINGFACE_API_TOKEN=your_hf_token_here
HUGGINGFACE_MODEL=google/flan-t5-base

# Database paths (optional - Railway uses ephemeral storage)
CHROMA_DB_PATH=./data/chroma
UPLOAD_DIR=./data/uploads

# API URL (will be set automatically by Railway, but you can override)
# API_URL will be your Railway app URL
```

**Note:** Railway automatically sets the `PORT` environment variable - you don't need to set it manually.

### 4. Deploy

Railway will automatically:
1. Detect it's a Python project (from `requirements.txt`)
2. Install dependencies
3. Run the command from `Procfile`
4. Start your FastAPI server

### 5. Access Your Application

1. Railway will provide you with a public URL (like `https://your-app.railway.app`)
2. Your API will be available at: `https://your-app.railway.app`
3. Test it: `https://your-app.railway.app/health`

## Important Notes

### Ephemeral Storage

⚠️ **Railway uses ephemeral storage**, which means:
- Files in `data/` and `chroma_db/` directories will be lost when the service restarts
- Consider using:
  - **Railway Volume** for persistent storage, OR
  - **External database** for ChromaDB (e.g., hosted ChromaDB or PostgreSQL with pgvector)

### Database Persistence Options

**Option 1: Use Railway Volume**
1. In Railway dashboard, add a "Volume" service
2. Mount it to `/app/data` or similar
3. Update `CHROMA_DB_PATH` to use the mounted volume path

**Option 2: Use External ChromaDB**
- Deploy ChromaDB separately or use a hosted service
- Update connection settings in your code

### Custom Domain

You can add a custom domain in Railway settings:
1. Go to your service settings
2. Click "Generate Domain" or add a custom domain

### Monitoring

- Check logs in the Railway dashboard
- Set up alerts for service health
- Monitor resource usage (CPU, Memory)

## Troubleshooting

### Build Fails
- Check that `requirements.txt` has all dependencies
- Ensure Python version in `runtime.txt` is supported

### App Crashes on Start
- Check logs in Railway dashboard
- Verify environment variables are set correctly
- Ensure PORT is not manually set (Railway sets it automatically)

### Database/Storage Issues
- Remember that Railway has ephemeral storage
- Use volumes or external storage for persistence

## Updating Your Deployment

Simply push changes to your Git repository, and Railway will automatically redeploy:

```bash
git add .
git commit -m "Update application"
git push
```

Railway will detect the changes and redeploy automatically.
